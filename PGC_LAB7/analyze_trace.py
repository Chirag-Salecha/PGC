import json
import sys
from collections import defaultdict

GAP_US = 50

def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged

def duration(intervals):
    return sum(end - start for start, end in intervals)

def overlap(a, b):
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        total += max(0, end - start)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total

def analyze(filename):
    with open(filename, encoding="utf-8") as file:
        data = json.load(file)

    events = data["traceEvents"] if isinstance(data, dict) else data

    kernels = []
    copies = []
    kernel_names = defaultdict(lambda: [0, 0.0])

    for event in events:
        if event.get("ph") != "X":
            continue

        category = event.get("cat", "")
        start = event.get("ts", 0)
        end = start + event.get("dur", 0)

        if category == "kernel":
            kernels.append([start, end])
            name = event.get("name", "Unknown")
            kernel_names[name][0] += 1
            kernel_names[name][1] += end - start

        elif category in ("gpu_memcpy", "gpu_memset"):
            copies.append([start, end])

    if not kernels:
        print("No GPU kernels found in the trace.")
        return

    k = merge_intervals(kernels)
    c = merge_intervals(copies)

    first = min(k[0][0], c[0][0] if c else k[0][0])
    last = max(k[-1][1], c[-1][1] if c else k[-1][1])

    span = last - first
    busy = duration(k)
    gaps = [
        k[i + 1][0] - k[i][1]
        for i in range(len(k) - 1)
    ]
    large_gaps = [gap for gap in gaps if gap > GAP_US]

    print("=" * 60)
    print("EXPERIMENT 7: BASELINE GPU PROFILING REPORT")
    print("=" * 60)
    print(f"Profiled window       : {span / 1000:.3f} ms")
    print(f"GPU kernel busy time  : {busy / 1000:.3f} ms")
    print(f"Kernel busy fraction  : {100 * busy / span:.2f} %")
    print(f"Memory copy time      : {duration(c) / 1000:.3f} ms")
    print(f"Copy/compute overlap  : {overlap(k, c) / 1000:.3f} ms")
    print(f"Idle gaps > 50 us     : {len(large_gaps)}")
    print(f"Large idle gap time   : {sum(large_gaps) / 1000:.3f} ms")

    print("\nGPU KERNEL DETAILS")
    print("-" * 60)

    for name, (calls, elapsed) in sorted(
        kernel_names.items(),
        key=lambda item: -item[1][1]
    ):
        kind = (
            "Compute-bound candidate"
            if any(x in name.lower() for x in
                   ("gemm", "sgemm", "cublas", "cutlass"))
            else "Memory-bound candidate"
        )
        print(f"{calls:4d} calls | {elapsed / 1000:8.3f} ms")
        print(f"  {kind}: {name[:90]}")

    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    analyze(sys.argv[1])
