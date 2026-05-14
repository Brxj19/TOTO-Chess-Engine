#!/usr/bin/env python3
"""Benchmark Stockfish and TCE-owned NNUE backends through UCI."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


STARTPOS_FEN = "rn1qkbnr/ppp1pppp/8/3p4/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 1 3"


@dataclass
class BenchRow:
    backend: str
    fen: str
    depth: int
    bestmove: str
    score_cp: str
    nodes: str
    time_ms: str
    nps: str
    pv: str


class UciSession:
    def __init__(self, engine: str, timeout: float) -> None:
        self.timeout = timeout
        self.output: list[str] = []
        self.process = subprocess.Popen(
            [engine],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("engine stdin is closed")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, token: str) -> str:
        if self.process.stdout is None:
            raise RuntimeError("engine stdout is closed")

        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                remaining = self.process.stdout.read() or ""
                if remaining:
                    self.output.append(remaining)
                raise RuntimeError(f"engine exited before {token!r}")

            line = self.process.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue

            self.output.append(line)
            if token in line or token in "".join(self.output):
                return "".join(self.output)

        raise TimeoutError(f"timed out waiting for {token!r}")

    def quit(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
            except (BrokenPipeError, RuntimeError):
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TCE evaluation backends.")
    parser.add_argument("--engine", default="./tce", help="Engine command path.")
    parser.add_argument(
        "--tcennue",
        default="data/nnue_runs/baseline/tce_baseline.tcennue",
        help="TCE-owned .tcennue file.",
    )
    parser.add_argument("--depth", type=int, default=4, help="Search depth.")
    parser.add_argument(
        "--positions",
        type=Path,
        default=Path("tools/nnue_train/test_vectors/benchmark_positions.fen"),
        help="Input FEN list. Lines may contain 'startpos'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/nnue_runs/backend_benchmark.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout seconds.")
    return parser.parse_args()


def load_positions(path: Path) -> list[str]:
    positions = []
    with path.open("r", encoding="utf-8") as fen_file:
        for line in fen_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            positions.append(line)
    if not positions:
        raise ValueError(f"{path} did not contain any positions")
    return positions


def position_command(fen: str) -> str:
    if fen == "startpos":
        return "position startpos"
    return f"position fen {fen}"


def parse_info(output: str) -> tuple[str, str, str, str, str]:
    info_lines = [line.strip() for line in output.splitlines() if line.startswith("info ")]
    if not info_lines:
        return "", "", "", "", ""

    final = info_lines[-1]
    score_cp = ""
    nodes = ""
    time_ms = ""
    nps = ""
    pv = ""

    score_match = re.search(r"\bscore cp (-?\d+)", final)
    if score_match:
        score_cp = score_match.group(1)
    else:
        mate_match = re.search(r"\bscore mate (-?\d+)", final)
        if mate_match:
            score_cp = f"mate {mate_match.group(1)}"

    nodes_match = re.search(r"\bnodes (\d+)", final)
    if nodes_match:
        nodes = nodes_match.group(1)

    time_match = re.search(r"\btime (\d+)", final)
    if time_match:
        time_ms = time_match.group(1)

    nps_match = re.search(r"\bnps (\d+)", final)
    if nps_match:
        nps = nps_match.group(1)
    elif nodes and time_ms and int(time_ms) > 0:
        nps = str(int(int(nodes) * 1000 / int(time_ms)))

    pv_match = re.search(r"\bpv (.+)$", final)
    if pv_match:
        pv = pv_match.group(1)

    return score_cp, nodes, time_ms, nps, pv


def parse_bestmove(output: str) -> str:
    for line in reversed(output.splitlines()):
        if line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else ""
    return ""


def run_position(
    engine: str,
    backend: str,
    tcennue: str,
    fen: str,
    depth: int,
    timeout: float,
) -> BenchRow:
    session = UciSession(engine, timeout)
    try:
        session.send("uci")
        session.read_until("uciok")
        session.send(f"setoption name EvalBackend value {backend}")
        if backend == "tce":
            session.read_until("EvalBackend set to tce")
            session.send(f"setoption name EvalFile value {tcennue}")
            session.read_until("loaded TCE NNUE")
        session.send("isready")
        session.read_until("readyok")
        session.send(position_command(fen))
        session.send(f"go depth {depth}")
        output = session.read_until("bestmove")
    finally:
        session.quit()

    score_cp, nodes, time_ms, nps, pv = parse_info(output)
    return BenchRow(
        backend=backend,
        fen=fen,
        depth=depth,
        bestmove=parse_bestmove(output),
        score_cp=score_cp,
        nodes=nodes,
        time_ms=time_ms,
        nps=nps,
        pv=pv,
    )


def write_csv(path: Path, rows: list[BenchRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(BenchRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def print_summary(rows: list[BenchRow]) -> None:
    print("backend   depth  score_cp  nodes    time_ms  nps      bestmove  fen")
    print("-------   -----  --------  -----    -------  ---      --------  ---")
    for row in rows:
        fen_label = row.fen if row.fen == "startpos" else row.fen[:36] + "..."
        print(
            f"{row.backend:<9} {row.depth:<5} {row.score_cp:<8} "
            f"{row.nodes:<8} {row.time_ms:<8} {row.nps:<8} "
            f"{row.bestmove:<8} {fen_label}"
        )


def main() -> int:
    args = parse_args()
    if args.depth < 1:
        print("--depth must be at least 1", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    positions = load_positions(args.positions)
    rows: list[BenchRow] = []
    for fen in positions:
        for backend in ("stockfish", "tce"):
            rows.append(
                run_position(
                    args.engine,
                    backend,
                    args.tcennue,
                    fen,
                    args.depth,
                    args.timeout,
                )
            )

    write_csv(args.output, rows)
    print_summary(rows)
    print(f"\nwrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
