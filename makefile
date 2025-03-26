# Source files
SRC = src/toto.c

# Output executables
NATIVE_EXE = tce
WIN64_EXE = toto.exe

# Default target
all: $(NATIVE_EXE) $(WIN64_EXE)

# Build native executable
$(NATIVE_EXE): $(SRC)
	$(CC) -Ofast $(SRC) -o $(NATIVE_EXE)

# Build Windows 64-bit executable
$(WIN64_EXE): $(SRC)
	x86_64-w64-mingw32-gcc -Ofast $(SRC) -o $(WIN64_EXE)

# Debug target
debug: $(SRC)
	$(CC) $(SRC) -o $(NATIVE_EXE)
	x86_64-w64-mingw32-gcc $(SRC) -o $(WIN64_EXE)