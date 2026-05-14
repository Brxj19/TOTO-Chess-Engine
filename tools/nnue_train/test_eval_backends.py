#!/usr/bin/env python3
"""Regression test the Stockfish and optional TCE NNUE UCI backends."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass
class TestResult:
    name: str
    passed: bool
    output: str
    error: str | None = None


class UciSession:
    def __init__(self, engine: str, timeout: float) -> None:
        self.engine = engine
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

    def read_until(self, token: str, timeout: float | None = None) -> str:
        if self.process.stdout is None:
            raise RuntimeError("engine stdout is closed")

        deadline = time.monotonic() + (timeout if timeout is not None else self.timeout)
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                remaining = self.process.stdout.read() or ""
                if remaining:
                    self.output.append(remaining)
                raise RuntimeError(f"engine exited before token {token!r}")

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

    def captured_output(self) -> str:
        return "".join(self.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TCE UCI evaluation backends.")
    parser.add_argument("--engine", default="./tce", help="Engine command path.")
    parser.add_argument(
        "--tcennue",
        default="data/nnue_runs/baseline/tce_baseline.tcennue",
        help="TCE-owned .tcennue file for the optional backend.",
    )
    parser.add_argument("--depth", type=int, default=2, help="Search depth for tests.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout in seconds.")
    return parser.parse_args()


def run_uci_session(
    name: str,
    engine: str,
    commands: list[tuple[str, str | None]],
    expected_tokens: list[str],
    timeout: float,
) -> TestResult:
    session = UciSession(engine, timeout)
    try:
        for command, wait_for in commands:
            session.send(command)
            if wait_for:
                session.read_until(wait_for)

        output = session.captured_output()
        for token in expected_tokens:
            if token not in output:
                raise RuntimeError(f"missing expected token {token!r}")

        return TestResult(name=name, passed=True, output=output)
    except Exception as exc:
        return TestResult(
            name=name,
            passed=False,
            output=session.captured_output(),
            error=str(exc),
        )
    finally:
        session.quit()


def print_result(result: TestResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: {result.name}")
    if not result.passed:
        print(f"error: {result.error}")
        print("captured engine output:")
        print(result.output.rstrip())


def main() -> int:
    args = parse_args()
    if args.depth < 1:
        print("--depth must be at least 1", file=sys.stderr)
        return 2
    if args.timeout <= 0:
        print("--timeout must be positive", file=sys.stderr)
        return 2

    tests = [
        (
            "default Stockfish backend",
            [
                ("uci", "uciok"),
                ("isready", "readyok"),
                ("position startpos", None),
                (f"go depth {args.depth}", "bestmove"),
            ],
            ["uciok", "readyok", "bestmove"],
        ),
        (
            "TCE-owned NNUE backend",
            [
                ("uci", "uciok"),
                ("setoption name EvalBackend value tce", "EvalBackend set to tce"),
                (f"setoption name EvalFile value {args.tcennue}", "loaded TCE NNUE"),
                ("isready", "readyok"),
                ("position startpos", None),
                (f"go depth {args.depth}", "bestmove"),
            ],
            ["uciok", "loaded TCE NNUE", "readyok", "bestmove"],
        ),
        (
            "invalid TCE NNUE fallback",
            [
                ("uci", "uciok"),
                ("setoption name EvalBackend value tce", "EvalBackend set to tce"),
                ("setoption name EvalFile value wrong/path.tcennue", "failed to load TCE NNUE"),
                ("isready", "readyok"),
                ("position startpos", None),
                ("go depth 1", "bestmove"),
            ],
            ["uciok", "failed to load TCE NNUE", "readyok", "bestmove"],
        ),
    ]

    results = [
        run_uci_session(name, args.engine, commands, tokens, args.timeout)
        for name, commands, tokens in tests
    ]

    for result in results:
        print_result(result)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
