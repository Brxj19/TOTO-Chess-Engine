# TOTO Chess Engine

TOTO is a chess engine implemented in C. This project aims to provide a robust and efficient representation of a chessboard, along with the necessary logic to play and evaluate chess games.

## Project Structure

```
TOTO
├── src
│   ├── toto.c        # Implementation of toto engine
├── tests
│   ├── test_toto.cpp    # Unit tests for the Board class
├── CMakeLists.txt       # CMake configuration file
└── README.md            # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/TOTO.git
   cd TOTO
   ```

2. **Build the project using CMake:**
   ```
   mkdir build
   cd build
   cmake ..
   make
   ```

3. **Run the application:**
   ```
   ./TOTO
   ```

## Usage

- The chess engine can be run from the command line. Follow the prompts to make moves and interact with the engine.

## Testing

- To run the unit tests, navigate to the `build` directory and execute:
   ```
   make test
   ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.