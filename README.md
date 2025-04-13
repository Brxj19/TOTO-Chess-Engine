# TOTO Chess Engine

TOTO is a chess engine implemented in C. This project aims to provide a robust and efficient representation of a chessboard, along with the necessary logic to play and evaluate chess games.

## Project Structure

```
TOTO
├── nnue
│   ├── misc.cpp              #stockfish nnue utils implementations
│   ├── nnue.cpp              #stockfish nnue evaluation implementations
│   ├── misc.h                #stockfish nnue utils header
│   ├── nnue.h                #stockfish nnue evaluation header
├── makefile                  # Make configuration file
├── toto.c                    # toto chess engine file
├── nnue_eval.c               #stockfish nnue wrapper implementations
├── nnue_eval.                #stockfish nnue wrapper header
├── nn-eba324f53044.nnue      #stockfish nnue file
└── README.md                 # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/Brxj19/TOTO-Chess-Engine.git         
   cd TOTO-Chess-Engine
   ```

2. **Build the project using Make:**
   ```
   make clear
   make
   ```

3. **Run the application:**
   ```
   ./tce
   ```

## Usage
  
- The chess engine can be run from the command line. Follow the prompts to make moves and interact with the engine.

1. **uci command**
   ```
   uci
   ```
2. **isready command**
   ```
   isready
   ```
3. **ucinewgame command**
   ```
   ucinewgame
   ```
4. **position command**
   ```
   position startpos
   ```
5. **moves command**
   ```
   position startpos moves e2e4
   position startpos moves e2e4 e7e5
   ```
6. **go command**
   ```
   go depth 8
   ```
## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
