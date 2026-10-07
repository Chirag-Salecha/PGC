import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Custom typography & theme settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Segoe UI', 'Arial', 'Helvetica']

# Benchmark Data (Workload: N = 1,000,000,000 iterations)
threads = [1, 2, 4, 6, 16]
seq_time = 3.346414

pthread_times = [3.205682, 1.689293, 0.947512, 0.850071, 0.530415]
omp_times     = [3.186103, 1.662809, 0.978149, 0.841213, 0.448971]

pthread_speedup = [seq_time / t for t in pthread_times]
omp_speedup     = [seq_time / t for t in omp_times]

pthread_eff = [(s / t) * 100 for s, t in zip(pthread_speedup, threads)]
omp_eff     = [(s / t) * 100 for s, t in zip(omp_speedup, threads)]

# Sophisticated Design Palette
COLOR_PTHREAD = '#2563EB'   # Vibrant Royal Blue
COLOR_OMP     = '#0D9488'   # Deep Teal
COLOR_SEQ     = '#DC2626'   # High-Contrast Crimson
COLOR_IDEAL   = '#64748B'   # Slate Gray
COLOR_BG      = '#F8FAFC'   # Soft modern slate-white background
COLOR_CARD    = '#FFFFFF'
COLOR_TEXT    = '#0F172A'   # Deep Slate 900
COLOR_SUBTEXT = '#475569'   # Slate 600
COLOR_GRID    = '#E2E8F0'

HATCH_PTHREAD = '///'
HATCH_OMP     = '\\\\'

x = np.arange(len(threads))
width = 0.35

# ==========================================
# 1. STANDALONE EXECUTION TIME CHART
# ==========================================
fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=COLOR_BG)
ax.set_facecolor(COLOR_CARD)

bars1 = ax.bar(x - width/2, pthread_times, width, label='POSIX Threads (Pthreads)',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.2, hatch=HATCH_PTHREAD, alpha=0.95)
bars2 = ax.bar(x + width/2, omp_times, width, label='OpenMP Runtime',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.2, hatch=HATCH_OMP, alpha=0.95)

# Sequential Baseline line
line_seq = ax.axhline(seq_time, color=COLOR_SEQ, linestyle='--', linewidth=2.0,
                      label=f'Sequential Baseline ({seq_time:.3f} s)')

ax.set_title('Execution Time vs. Number of Threads (Lower is Better)',
             fontsize=14, fontweight='bold', color=COLOR_TEXT, pad=18)
ax.set_xlabel('Number of Worker Threads', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_ylabel('Execution Time in Seconds', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels([f'{t} Thread{"s" if t > 1 else ""}' for t in threads], fontsize=10.5, fontweight='bold', color=COLOR_TEXT)
ax.set_ylim(0, 4.2)

ax.grid(True, linestyle=':', color=COLOR_GRID, linewidth=1.2, alpha=0.85, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.legend(frameon=True, facecolor='#F1F5F9', edgecolor='#CBD5E1', fontsize=10, loc='upper right')

for bar in bars1:
    h = bar.get_height()
    offset = -0.45 if h > 3.0 else 0.08
    txt_color = '#FFFFFF' if offset < 0 else COLOR_TEXT
    bg_color  = '#1E3A8A' if offset < 0 else '#EFF6FF'
    bdr_color = '#172554' if offset < 0 else '#BFDBFE'
    ax.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}s',
            ha='center', va='bottom' if offset > 0 else 'top', fontsize=9.5, fontweight='bold', color=txt_color,
            bbox=dict(boxstyle='round,pad=0.25', fc=bg_color, ec=bdr_color, lw=0.9))

for bar in bars2:
    h = bar.get_height()
    offset = -0.45 if h > 3.0 else 0.08
    txt_color = '#FFFFFF' if offset < 0 else COLOR_TEXT
    bg_color  = '#115E59' if offset < 0 else '#F0FDFA'
    bdr_color = '#042F2E' if offset < 0 else '#99F6E4'
    ax.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}s',
            ha='center', va='bottom' if offset > 0 else 'top', fontsize=9.5, fontweight='bold', color=txt_color,
            bbox=dict(boxstyle='round,pad=0.25', fc=bg_color, ec=bdr_color, lw=0.9))

plt.tight_layout()
plt.savefig('images/execution_time_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 2. STANDALONE SPEEDUP CHART
# ==========================================
fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=COLOR_BG)
ax.set_facecolor(COLOR_CARD)

bars1 = ax.bar(x - width/2, pthread_speedup, width, label='Pthreads Measured Speedup',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.2, hatch=HATCH_PTHREAD, alpha=0.95)
bars2 = ax.bar(x + width/2, omp_speedup, width, label='OpenMP Measured Speedup',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.2, hatch=HATCH_OMP, alpha=0.95)

# Reference points for linear scaling comparison
ideal_y = [1, 2, 4, 6, 16]
ax.plot(x, ideal_y, color=COLOR_IDEAL, linestyle='--', marker='s', markersize=5,
        linewidth=1.8, label='Theoretical Linear Speedup (Ideal S = p)')

ax.set_title('Parallel Speedup Factor relative to Sequential Baseline',
             fontsize=14, fontweight='bold', color=COLOR_TEXT, pad=18)
ax.set_xlabel('Number of Worker Threads', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_ylabel('Speedup Ratio (Higher is Better)', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels([f'{t} Thread{"s" if t > 1 else ""}' for t in threads], fontsize=10.5, fontweight='bold', color=COLOR_TEXT)
ax.set_ylim(0, 18.0)

ax.grid(True, linestyle=':', color=COLOR_GRID, linewidth=1.2, alpha=0.85, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.legend(frameon=True, facecolor='#F1F5F9', edgecolor='#CBD5E1', fontsize=10, loc='upper left')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.35, f'{h:.2f}x',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.25', fc='#EFF6FF', ec='#BFDBFE', lw=0.9))

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.35, f'{h:.2f}x',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.25', fc='#F0FDFA', ec='#99F6E4', lw=0.9))

plt.tight_layout()
plt.savefig('images/speedup_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 3. STANDALONE EFFICIENCY CHART
# ==========================================
fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=COLOR_BG)
ax.set_facecolor(COLOR_CARD)

bars1 = ax.bar(x - width/2, pthread_eff, width, label='Pthreads Parallel Efficiency',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.2, hatch=HATCH_PTHREAD, alpha=0.95)
bars2 = ax.bar(x + width/2, omp_eff, width, label='OpenMP Parallel Efficiency',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.2, hatch=HATCH_OMP, alpha=0.95)

ax.axhline(100.0, color=COLOR_IDEAL, linestyle='--', linewidth=1.8, label='100% Ideal Efficiency Line')

ax.set_title('Parallel Efficiency vs. Thread Count',
             fontsize=14, fontweight='bold', color=COLOR_TEXT, pad=18)
ax.set_xlabel('Number of Worker Threads', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_ylabel('Parallel Efficiency (%) — Higher is Better', fontsize=11, fontweight='bold', color=COLOR_SUBTEXT, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels([f'{t} Thread{"s" if t > 1 else ""}' for t in threads], fontsize=10.5, fontweight='bold', color=COLOR_TEXT)
ax.set_ylim(0, 125)

ax.grid(True, linestyle=':', color=COLOR_GRID, linewidth=1.2, alpha=0.85, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.legend(frameon=True, facecolor='#F1F5F9', edgecolor='#CBD5E1', fontsize=10, loc='upper right')

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 2.0, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.25', fc='#EFF6FF', ec='#BFDBFE', lw=0.9))

for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 2.0, f'{h:.1f}%',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=COLOR_TEXT,
            bbox=dict(boxstyle='round,pad=0.25', fc='#F0FDFA', ec='#99F6E4', lw=0.9))

plt.tight_layout()
plt.savefig('images/efficiency_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# ==========================================
# 4. COMBINED PERFORMANCE DASHBOARD (3-PANEL)
# ==========================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.0), facecolor=COLOR_BG)

# (A) Execution Time
ax1.set_facecolor(COLOR_CARD)
b1_1 = ax1.bar(x - width/2, pthread_times, width, label='Pthreads',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.1, hatch=HATCH_PTHREAD)
b1_2 = ax1.bar(x + width/2, omp_times, width, label='OpenMP',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.1, hatch=HATCH_OMP)
ax1.axhline(seq_time, color=COLOR_SEQ, linestyle='--', linewidth=1.8, label='Sequential')
ax1.set_title('(A) Execution Time (s)', fontsize=12.5, fontweight='bold', color=COLOR_TEXT, pad=14)
ax1.set_xlabel('Threads', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax1.set_ylabel('Seconds (Lower is Better)', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax1.set_xticks(x)
ax1.set_xticklabels(threads, fontsize=10, fontweight='bold')
ax1.set_ylim(0, 4.3)
ax1.grid(True, linestyle=':', color=COLOR_GRID, alpha=0.85, axis='y')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(fontsize=9, loc='upper right', framealpha=0.9)

# Panel 1: Execution Time
for bar in b1_1:
    h = bar.get_height()
    offset = -0.5 if h > 3.0 else 0.08
    txt_color = '#FFFFFF' if offset < 0 else COLOR_TEXT
    ax1.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}s',
             ha='center', va='bottom' if offset > 0 else 'top', fontsize=8.0, fontweight='bold', color=txt_color)
for bar in b1_2:
    h = bar.get_height()
    offset = -0.5 if h > 3.0 else 0.08
    txt_color = '#FFFFFF' if offset < 0 else COLOR_TEXT
    ax1.text(bar.get_x() + bar.get_width()/2., h + offset, f'{h:.2f}s',
             ha='center', va='bottom' if offset > 0 else 'top', fontsize=8.0, fontweight='bold', color=txt_color)

# (B) Speedup
ax2.set_facecolor(COLOR_CARD)
b2_1 = ax2.bar(x - width/2, pthread_speedup, width, label='Pthreads',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.1, hatch=HATCH_PTHREAD)
b2_2 = ax2.bar(x + width/2, omp_speedup, width, label='OpenMP',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.1, hatch=HATCH_OMP)
ax2.plot(x, ideal_y, color=COLOR_IDEAL, linestyle='--', marker='s', markersize=4, label='Linear Ref')
ax2.set_title('(B) Speedup Factor (x)', fontsize=12.5, fontweight='bold', color=COLOR_TEXT, pad=14)
ax2.set_xlabel('Threads', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax2.set_ylabel('Speedup Ratio (Higher is Better)', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax2.set_xticks(x)
ax2.set_xticklabels(threads, fontsize=10, fontweight='bold')
ax2.set_ylim(0, 18.0)
ax2.grid(True, linestyle=':', color=COLOR_GRID, alpha=0.85, axis='y')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)

for bar in b2_1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.35, f'{h:.2f}x', ha='center', va='bottom', fontsize=7.8, fontweight='bold')
for bar in b2_2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.35, f'{h:.2f}x', ha='center', va='bottom', fontsize=7.8, fontweight='bold')

# (C) Efficiency
ax3.set_facecolor(COLOR_CARD)
b3_1 = ax3.bar(x - width/2, pthread_eff, width, label='Pthreads',
               color=COLOR_PTHREAD, edgecolor='#1D4ED8', linewidth=1.1, hatch=HATCH_PTHREAD)
b3_2 = ax3.bar(x + width/2, omp_eff, width, label='OpenMP',
               color=COLOR_OMP, edgecolor='#0F766E', linewidth=1.1, hatch=HATCH_OMP)
ax3.axhline(100.0, color=COLOR_IDEAL, linestyle='--', linewidth=1.8, label='100% Ideal')
ax3.set_title('(C) Parallel Efficiency (%)', fontsize=12.5, fontweight='bold', color=COLOR_TEXT, pad=14)
ax3.set_xlabel('Threads', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax3.set_ylabel('Efficiency % (Higher is Better)', fontsize=10.5, fontweight='bold', color=COLOR_SUBTEXT)
ax3.set_xticks(x)
ax3.set_xticklabels(threads, fontsize=10, fontweight='bold')
ax3.set_ylim(0, 125)
ax3.grid(True, linestyle=':', color=COLOR_GRID, alpha=0.85, axis='y')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(fontsize=9, loc='upper right', framealpha=0.9)

for bar in b3_1:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 2.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.8, fontweight='bold')
for bar in b3_2:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 2.0, f'{h:.1f}%', ha='center', va='bottom', fontsize=7.8, fontweight='bold')

plt.suptitle('Parallel & Distributed Systems — Experiment 2: POSIX Threads vs. OpenMP Benchmark Analysis',
             fontsize=15, fontweight='bold', color=COLOR_TEXT, y=1.02)
plt.tight_layout()
plt.savefig('images/performance_comparison_charts.png', dpi=300, bbox_inches='tight')
plt.close()

print('Charts re-rendered successfully with enhanced layout and typography.')
