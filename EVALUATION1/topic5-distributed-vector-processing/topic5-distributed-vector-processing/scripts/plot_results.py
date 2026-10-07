#!/usr/bin/env python3
"""
==============================================================================
Performance Analysis & Graph Generator
File: scripts/plot_results.py
Topic: 5 (Distributed Vector Processing)

Reads benchmark results from results/timing_results.csv and generates
high-resolution performance plots for:
  1. Execution Time vs Number of Processes
  2. Speedup vs Number of Processes (with Ideal Linear Speedup)
  3. Parallel Efficiency vs Number of Processes (with Ideal 100% Line)
  4. Combined Summary Dashboard
==============================================================================
"""

import os
import sys
import csv

def generate_plots():
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg') # Headless backend
    except ImportError:
        print("[Notice] matplotlib not found. Installing via pip or system...")
        os.system("python3 -m pip install matplotlib --quiet")
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    csv_file = os.path.join(root_dir, "results", "timing_results.csv")
    graphs_dir = os.path.join(root_dir, "graphs")

    if not os.path.exists(csv_file):
        print(f"Error: Timing results CSV not found at {csv_file}")
        print("Please run scripts/run_experiments.sh first.")
        sys.exit(1)

    os.makedirs(graphs_dir, exist_ok=True)

    # Read and group data by Vector Size N
    # Columns: Size_N,Processes,Seq_Time,Parallel_Time,Comp_Time,Comm_Time,Speedup,Efficiency
    data_by_size = {}

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            n = int(row['Size_N'])
            p = int(row['Processes'])
            seq_t = float(row['Seq_Time'])
            par_t = float(row['Parallel_Time'])
            comp_t = float(row['Comp_Time'])
            comm_t = float(row['Comm_Time'])
            speedup = float(row['Speedup'])
            eff = float(row['Efficiency'])

            if n not in data_by_size:
                data_by_size[n] = []
            data_by_size[n].append({
                'p': p,
                'seq_t': seq_t,
                'par_t': par_t,
                'comp_t': comp_t,
                'comm_t': comm_t,
                'speedup': speedup,
                'eff': eff
            })

    # Styling settings
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    # -------------------------------------------------------------------------
    # Graph 1: Execution Time vs Number of Processes
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    for idx, (n, records) in enumerate(data_by_size.items()):
        procs = [r['p'] for r in records]
        par_times = [r['par_t'] for r in records]
        label_text = f"N = {n:,} elements"
        plt.plot(procs, par_times, marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)], linewidth=2, markersize=7,
                 label=label_text)

    plt.title("Execution Time vs. Number of Processes (Topic 5: MPI)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Processes (P)", fontsize=11)
    plt.ylabel("Execution Time (seconds)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    plt.tight_layout()
    time_plot_path = os.path.join(graphs_dir, "execution_time.png")
    plt.savefig(time_plot_path)
    plt.close()
    print(f"Saved: {time_plot_path}")

    # -------------------------------------------------------------------------
    # Graph 2: Speedup vs Number of Processes
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    all_procs = sorted(list({r['p'] for records in data_by_size.values() for r in records}))
    # Ideal linear speedup line
    plt.plot(all_procs, all_procs, 'k--', linewidth=1.5, label='Ideal Linear Speedup (S=P)')

    for idx, (n, records) in enumerate(data_by_size.items()):
        procs = [r['p'] for r in records]
        speedups = [r['speedup'] for r in records]
        label_text = f"Actual Speedup (N = {n:,})"
        plt.plot(procs, speedups, marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)], linewidth=2, markersize=7,
                 label=label_text)

    plt.title("Parallel Speedup vs. Number of Processes", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Processes (P)", fontsize=11)
    plt.ylabel("Speedup ($S_p = T_{seq} / T_{par}$)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    plt.tight_layout()
    speedup_plot_path = os.path.join(graphs_dir, "speedup.png")
    plt.savefig(speedup_plot_path)
    plt.close()
    print(f"Saved: {speedup_plot_path}")

    # -------------------------------------------------------------------------
    # Graph 3: Efficiency vs Number of Processes
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    # Ideal 100% Efficiency
    plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal Efficiency (1.0 / 100%)')

    for idx, (n, records) in enumerate(data_by_size.items()):
        procs = [r['p'] for r in records]
        efficiencies = [r['eff'] for r in records]
        label_text = f"Efficiency (N = {n:,})"
        plt.plot(procs, efficiencies, marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)], linewidth=2, markersize=7,
                 label=label_text)

    plt.title("Parallel Efficiency vs. Number of Processes", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Number of Processes (P)", fontsize=11)
    plt.ylabel("Efficiency ($E_p = S_p / P$)", fontsize=11)
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, 1.15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    plt.tight_layout()
    efficiency_plot_path = os.path.join(graphs_dir, "efficiency.png")
    plt.savefig(efficiency_plot_path)
    plt.close()
    print(f"Saved: {efficiency_plot_path}")

    # -------------------------------------------------------------------------
    # Graph 4: Combined Dashboard
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Subplot 1: Execution Time
    for idx, (n, records) in enumerate(data_by_size.items()):
        axs[0].plot([r['p'] for r in records], [r['par_t'] for r in records],
                    marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                    linewidth=2, label=f"N={n:,}")
    axs[0].set_title("Execution Time vs. Processes", fontweight='bold')
    axs[0].set_xlabel("Processes (P)")
    axs[0].set_ylabel("Time (seconds)")
    axs[0].set_xticks([1, 2, 4, 8])
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    # Subplot 2: Speedup
    axs[1].plot(all_procs, all_procs, 'k--', label='Ideal (S=P)')
    for idx, (n, records) in enumerate(data_by_size.items()):
        axs[1].plot([r['p'] for r in records], [r['speedup'] for r in records],
                    marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                    linewidth=2, label=f"N={n:,}")
    axs[1].set_title("Speedup vs. Processes", fontweight='bold')
    axs[1].set_xlabel("Processes (P)")
    axs[1].set_ylabel("Speedup ($S_p$)")
    axs[1].set_xticks([1, 2, 4, 8])
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    # Subplot 3: Efficiency
    axs[2].axhline(y=1.0, color='gray', linestyle='--', label='Ideal (1.0)')
    for idx, (n, records) in enumerate(data_by_size.items()):
        axs[2].plot([r['p'] for r in records], [r['eff'] for r in records],
                    marker=markers[idx % len(markers)], color=colors[idx % len(colors)],
                    linewidth=2, label=f"N={n:,}")
    axs[2].set_title("Parallel Efficiency vs. Processes", fontweight='bold')
    axs[2].set_xlabel("Processes (P)")
    axs[2].set_ylabel("Efficiency ($E_p$)")
    axs[2].set_xticks([1, 2, 4, 8])
    axs[2].set_ylim(0, 1.15)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()

    plt.suptitle("Topic 5: Distributed Vector Processing - Performance Evaluation",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    summary_plot_path = os.path.join(graphs_dir, "performance_summary.png")
    plt.savefig(summary_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {summary_plot_path}")
    print("All plots generated successfully!")

if __name__ == '__main__':
    generate_plots()
