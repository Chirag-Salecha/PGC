import subprocess
import sys
import re
import csv
import math
import os

os.makedirs("RESULTS", exist_ok=True)

results = {}

for mode in ["baseline", "optimized"]:
    print(f"\nRunning {mode.upper()} on local NVIDIA GPU...")

    p = subprocess.run(
        [sys.executable, "gpu_workload.py", "--mode", mode],
        capture_output=True,
        text=True,
        check=True
    )

    print(p.stdout)

    def extract(label):
        match = re.search(
            rf"{re.escape(label)}:\s*([\d.eE+-]+)",
            p.stdout
        )
        if not match:
            raise ValueError(f"Missing {label}")
        return float(match.group(1))

    results[mode] = {
        "time": extract("Total time"),
        "throughput": extract("Throughput"),
        "checksum": extract("Checksum")
    }

b = results["baseline"]
o = results["optimized"]

speedup = b["time"] / o["time"]
reduction = (b["time"] - o["time"]) / b["time"] * 100

print("\n" + "=" * 55)
print("EXPERIMENT 7 - LOCAL GPU PERFORMANCE COMPARISON")
print("=" * 55)
print(f"Baseline time     : {b['time']:.3f} seconds")
print(f"Optimized time    : {o['time']:.3f} seconds")
print(f"Baseline throughput  : {b['throughput']:.2f} batches/s")
print(f"Optimized throughput : {o['throughput']:.2f} batches/s")
print(f"Speedup           : {speedup:.2f}x")
print(f"Time reduction    : {reduction:.2f}%")

valid = math.isclose(
    b["checksum"], o["checksum"],
    rel_tol=1e-5
)

print("Checksum validation:", "MATCH" if valid else "MISMATCH")

with open("RESULTS/performance_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Mode", "Time (s)", "Throughput (batches/s)", "Checksum"])
    for mode, r in results.items():
        writer.writerow([mode, r["time"], r["throughput"], r["checksum"]])

print("\nSaved: RESULTS/performance_comparison.csv")
