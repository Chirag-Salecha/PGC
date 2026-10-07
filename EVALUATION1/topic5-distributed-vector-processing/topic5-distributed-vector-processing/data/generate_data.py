#!/usr/bin/env python3
"""
==============================================================================
Dataset Generator Script
File: data/generate_data.py
Topic: 5 (Distributed Vector Processing)

Generates benchmark vector datasets in text or binary format for testing,
offline verification, and external validation.
==============================================================================
"""

import argparse
import os
import struct

def generate_vector(size, output_file, mode="txt"):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    print(f"Generating vector of size N = {size:,} ({mode} mode) -> {output_file}...")

    if mode == "txt":
        with open(output_file, "w") as f:
            f.write(f"{size}\n")
            for i in range(size):
                val = (i + 1) * 0.001
                f.write(f"{val:.6f}\n")
    elif mode == "bin":
        with open(output_file, "wb") as f:
            f.write(struct.pack("q", size)) # 64-bit integer
            for i in range(size):
                val = (i + 1) * 0.001
                f.write(struct.pack("d", val)) # 64-bit double
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Successfully generated {output_file} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate vector data files.")
    parser.add_argument("--size", "-n", type=int, default=100000, help="Number of elements")
    parser.add_argument("--format", "-f", choices=["txt", "bin"], default="txt", help="File format")
    parser.add_argument("--out", "-o", type=str, default="data/sample_vector.txt", help="Output path")
    args = parser.parse_args()

    generate_vector(args.size, args.out, args.format)
