cmake_minimum_required(VERSION 3.10)

# Set the project name
project(TOTO C)

# Set the C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add the executable
add_executable(TOTO src/toto.c)