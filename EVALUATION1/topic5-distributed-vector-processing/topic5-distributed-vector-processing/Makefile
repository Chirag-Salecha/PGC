# ==============================================================================
# Makefile for Topic 5: Distributed Vector Processing
# Parallel Computing Mini-Project
# ==============================================================================

CC      = gcc
MPICC   = mpicc
CFLAGS  = -O3 -Wall -std=c99
LDFLAGS = -lm

SRC_DIR = src
BIN_SEQ = sequential
BIN_MPI = mpi_program

# Default target: compile both sequential and parallel binaries
all: $(BIN_SEQ) $(BIN_MPI)

# Compile sequential baseline
$(BIN_SEQ): $(SRC_DIR)/sequential.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "==> Sequential baseline compiled successfully: ./$(BIN_SEQ)"

# Compile MPI parallel program
$(BIN_MPI): $(SRC_DIR)/mpi_program.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDFLAGS)
	@echo "==> MPI program compiled successfully: ./$(BIN_MPI)"

# Quick run targets
run-seq: $(BIN_SEQ)
	./$(BIN_SEQ) 10000000

run-mpi: $(BIN_MPI)
	mpirun -np 4 ./$(BIN_MPI) 10000000

# Verification target: Compares sequential and MPI outputs bit-for-bit
test: all
	bash scripts/verify_correctness.sh

# Run complete benchmark experiments across processes and sizes
benchmark: all
	bash scripts/run_experiments.sh

# Generate graphs from collected results
graphs:
	python3 scripts/plot_results.py

# Clean generated binaries and temporary files
clean:
	rm -f $(BIN_SEQ) $(BIN_MPI)
	rm -f results/seq_out.txt results/mpi_out.txt
	@echo "==> Cleaned project binaries and temporary test files."
