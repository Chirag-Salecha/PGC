import csv
import os
import matplotlib.pyplot as plt

os.makedirs("RESULTS", exist_ok=True)

with open("RESULTS/performance_comparison.csv", newline="") as f:
    data = list(csv.DictReader(f))

if len(data) != 2:
    raise ValueError("Expected baseline and optimized results.")

by_mode = {row["Mode"].lower(): row for row in data}

baseline = by_mode["baseline"]
optimized = by_mode["optimized"]

modes = ["Baseline", "Optimized"]

times = [
    float(baseline["Time (s)"]),
    float(optimized["Time (s)"])
]

throughputs = [
    float(baseline["Throughput (batches/s)"]),
    float(optimized["Throughput (batches/s)"])
]

def make_graph(values, title, ylabel, filename, unit):
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(modes, values, width=0.45)

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.25)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f} {unit}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontweight="bold"
        )

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig("RESULTS/" + filename, dpi=300)
    plt.show()
    plt.close(fig)


make_graph(
    times,
    "RTX 5060 Ti - GPU Execution Time Comparison",
    "Execution Time (seconds)",
    "execution_time.png",
    "s"
)

make_graph(
    throughputs,
    "RTX 5060 Ti - GPU Throughput Comparison",
    "Throughput (batches/second)",
    "throughput.png",
    "batches/s"
)

speedup = times[0] / times[1]

print("=" * 55)
print("EXPERIMENT 7: PERFORMANCE SUMMARY")
print("=" * 55)
print(f"Baseline Time  : {times[0]:.3f} s")
print(f"Optimized Time : {times[1]:.3f} s")
print(f"Speedup        : {speedup:.2f}x")
print("Graphs saved successfully in RESULTS folder")
