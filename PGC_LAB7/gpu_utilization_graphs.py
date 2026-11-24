import csv
import re
import math
import os
import matplotlib.pyplot as plt

os.makedirs("RESULTS", exist_ok=True)

def read_gpu_log(mode):
    path = f"RESULTS/smi_{mode}.csv"

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = {h.strip(): h for h in reader.fieldnames}

        util_col = next(
            h for h in headers
            if "utilization.gpu" in h
        )

        mem_col = next(
            h for h in headers
            if "memory.used" in h
        )

        utilization = []
        memory = []

        for row in reader:
            def number(value):
                match = re.search(
                    r"\d+(?:\.\d+)?",
                    str(value)
                )
                return (
                    float(match.group())
                    if match else float("nan")
                )

            utilization.append(number(row[headers[util_col]]))
            memory.append(number(row[headers[mem_col]]))

    if not utilization:
        raise ValueError(f"No monitoring samples in {path}")

    return utilization, memory


baseline_util, baseline_mem = read_gpu_log("baseline")
optimized_util, optimized_mem = read_gpu_log("optimized")


def generate_graph(baseline, optimized, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        range(len(baseline)),
        baseline,
        linewidth=2,
        label="Baseline"
    )

    ax.plot(
        range(len(optimized)),
        optimized,
        linewidth=2,
        label="Optimized"
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Monitoring Sample Number (100 ms requested)")
    ax.set_ylabel(ylabel)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        f"RESULTS/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


generate_graph(
    baseline_util,
    optimized_util,
    "RTX 5060 Ti - GPU Utilization Comparison",
    "GPU Utilization (%)",
    "gpu_utilization_timeline.png"
)

generate_graph(
    baseline_mem,
    optimized_mem,
    "RTX 5060 Ti - GPU Memory Usage Comparison",
    "GPU Memory Used (MiB)",
    "gpu_memory_timeline.png"
)


def average_valid(values):
    valid = [x for x in values if math.isfinite(x)]
    return sum(valid) / len(valid) if valid else float("nan")


print("\n" + "=" * 60)
print("EXPERIMENT 7: GPU MONITORING SUMMARY")
print("=" * 60)

print(
    f"Baseline average GPU utilization: "
    f"{average_valid(baseline_util):.2f}%"
)

print(
    f"Optimized average GPU utilization: "
    f"{average_valid(optimized_util):.2f}%"
)

print(
    f"Baseline average GPU memory: "
    f"{average_valid(baseline_mem):.2f} MiB"
)

print(
    f"Optimized average GPU memory: "
    f"{average_valid(optimized_mem):.2f} MiB"
)

print("\nGraphs saved successfully in RESULTS folder.")
