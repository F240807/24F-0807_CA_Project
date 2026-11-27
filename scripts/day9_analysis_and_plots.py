#!/usr/bin/env python3
"""
Day 9 Analysis and Plotting Script
- Computes Amdahl's Law analysis across inference (BS=32) and training (Epoch 2).
- Exports data/amdahl_analysis.csv.
- Generates 5 publication-grade dark-theme plots in plots/ at 300 DPI (#131313 background).
"""

import os
import sys

# Ensure matplotlib uses a writable temp cache directory to avoid permissions warnings
os.environ['MPLCONFIGDIR'] = '/tmp/mplconfig'
os.makedirs('/tmp/mplconfig', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Global styling configuration for dark theme
BG_COLOR = '#131313'
PANEL_COLOR = '#1C1C1C'
GRID_COLOR = '#2A2A2A'
TEXT_COLOR = '#F0F0F0'
MUTED_TEXT = '#A0A0A0'

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'savefig.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'axes.edgecolor': '#444444',
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})

# Hardware Palette
COLORS = {
    'CPU FP32': '#FF6B6B',      # Coral Red
    'GPU FP32': '#4DABF7',      # Sky Blue
    'GPU FP16': '#748FFC',      # Indigo Blue
    'TPU FP32': '#FCC419',      # Warm Amber
    'TPU BF16': '#51CF66',      # Mint Green
    'Highlight': '#FF922B',    # Vivid Orange
    'Accent': '#CC5DE8',       # Purple
}

# ==============================================================================
# 1. AMDAHL'S LAW COMPUTATION & CSV EXPORT
# ==============================================================================
def compute_amdahl_analysis():
    print("[1/6] Computing Amdahl's Law analysis...")
    inf_cg = pd.read_csv('data/inference_cpu_gpu.csv')
    inf_tpu = pd.read_csv('data/inference_tpu.csv')
    tr_gpu = pd.read_csv('data/training_gpu.csv')
    tr_tpu = pd.read_csv('data/training_tpu.csv')

    records = []

    # Target workloads: BS=32 inference
    models = ['MobileViT-S', 'MobileNetV2', 'LSTM (2-Layer)']

    # GPU FP16 BS=32
    for m in models:
        row = inf_cg[(inf_cg['Model'] == m) & (inf_cg['Device'] == 'GPU') & 
                     (inf_cg['Precision'] == 'FP16') & (inf_cg['Batch_Size'] == 32)]
        if not row.empty:
            compute_ms = float(row['Steady_Median_ms'].values[0])
            e2e_ms = float(row['E2E_Median_ms'].values[0])
            records.append({
                'Workload': f'{m} Inference',
                'Device': 'GPU',
                'Precision': 'FP16',
                'Batch_Size_or_Epoch': 'BS=32',
                'Compute_ms': compute_ms,
                'E2E_ms': e2e_ms
            })

    # TPU FP32 & BF16 BS=32
    for prec in ['FP32', 'BF16']:
        for m in models:
            row = inf_tpu[(inf_tpu['Model'] == m) & (inf_tpu['Precision'] == prec) & 
                          (inf_tpu['Batch_Size'] == 32)]
            if not row.empty:
                compute_ms = float(row['Steady_Median_ms'].values[0])
                e2e_ms = float(row['E2E_Median_ms'].values[0])
                records.append({
                    'Workload': f'{m} Inference',
                    'Device': 'TPU',
                    'Precision': prec,
                    'Batch_Size_or_Epoch': 'BS=32',
                    'Compute_ms': compute_ms,
                    'E2E_ms': e2e_ms
                })

    # Training Epoch 2 (MobileViT-S) on GPU (FP32) and TPU (default FP32)
    # Steps = 64
    gpu_ep2 = tr_gpu[tr_gpu['Epoch'] == 2]
    if not gpu_ep2.empty:
        steps = float(gpu_ep2['Steps'].values[0])
        epoch_s = float(gpu_ep2['Epoch_Time_s'].values[0])
        e2e_step_ms = (epoch_s * 1000.0) / steps
        compute_step_ms = float(gpu_ep2['Steady_Median_Step_ms'].values[0])
        records.append({
            'Workload': 'MobileViT-S Training',
            'Device': 'GPU',
            'Precision': 'FP32',
            'Batch_Size_or_Epoch': 'Epoch 2',
            'Compute_ms': compute_step_ms,
            'E2E_ms': e2e_step_ms
        })

    tpu_ep2 = tr_tpu[tr_tpu['Epoch'] == 2]
    if not tpu_ep2.empty:
        steps = float(tpu_ep2['Steps'].values[0])
        epoch_s = float(tpu_ep2['Epoch_Time_s'].values[0])
        e2e_step_ms = (epoch_s * 1000.0) / steps
        compute_step_ms = float(tpu_ep2['Steady_Median_Step_ms'].values[0])
        records.append({
            'Workload': 'MobileViT-S Training',
            'Device': 'TPU',
            'Precision': 'FP32',
            'Batch_Size_or_Epoch': 'Epoch 2',
            'Compute_ms': compute_step_ms,
            'E2E_ms': e2e_step_ms
        })

    df = pd.DataFrame(records)
    df['Serial_Overhead_ms'] = df['E2E_ms'] - df['Compute_ms']
    df['Serial_Fraction_%'] = (df['Serial_Overhead_ms'] / df['E2E_ms']) * 100.0
    f_serial = df['Serial_Fraction_%'] / 100.0
    df['Speedup_Ceiling'] = 1.0 / f_serial
    # Speedup for 2x faster compute (s=2): S_2x = 1 / (f_serial + (1 - f_serial)/2)
    df['Speedup_2x_Compute'] = 1.0 / (f_serial + (1.0 - f_serial) / 2.0)

    # Format numeric columns for clean precision
    df['Compute_ms'] = df['Compute_ms'].round(3)
    df['E2E_ms'] = df['E2E_ms'].round(3)
    df['Serial_Overhead_ms'] = df['Serial_Overhead_ms'].round(3)
    df['Serial_Fraction_%'] = df['Serial_Fraction_%'].round(2)
    df['Speedup_Ceiling'] = df['Speedup_Ceiling'].round(3)
    df['Speedup_2x_Compute'] = df['Speedup_2x_Compute'].round(3)

    out_csv = 'data/amdahl_analysis.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} rows)")
    return df

# ==============================================================================
# 2. PLOT 1: Throughput vs Batch Size
# ==============================================================================
def plot_throughput_vs_batch_size():
    print("[2/6] Plotting throughput_vs_batch_size.png...")
    inf_cg = pd.read_csv('data/inference_cpu_gpu.csv')
    inf_tpu = pd.read_csv('data/inference_tpu.csv')

    models = ['MobileViT-S', 'MobileNetV2', 'LSTM (2-Layer)']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.suptitle('Inference Throughput vs. Batch Size Across Architectures', fontsize=16, weight='bold', y=0.98)

    configs = [
        ('CPU', 'FP32', 'CPU FP32', COLORS['CPU FP32'], 'o', '--'),
        ('GPU', 'FP32', 'GPU FP32', COLORS['GPU FP32'], 's', '-'),
        ('GPU', 'FP16', 'GPU FP16', COLORS['GPU FP16'], '^', '-'),
        ('TPU', 'FP32', 'TPU FP32', COLORS['TPU FP32'], 'D', '-.'),
        ('TPU', 'BF16', 'TPU BF16', COLORS['TPU BF16'], '*', '-')
    ]

    for idx, model in enumerate(models):
        ax = axes[idx]
        ax.set_title(f'{model}', pad=12)
        ax.set_xlabel('Batch Size', labelpad=8)
        if idx == 0:
            ax.set_ylabel('Throughput (samples/s)', labelpad=8)
        
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

        for dev, prec, label, color, marker, ls in configs:
            if dev in ['CPU', 'GPU']:
                sub = inf_cg[(inf_cg['Model'] == model) & (inf_cg['Device'] == dev) & (inf_cg['Precision'] == prec)]
            else:
                sub = inf_tpu[(inf_tpu['Model'] == model) & (inf_tpu['Precision'] == prec)]
            
            if not sub.empty:
                sub = sub.sort_values('Batch_Size')
                ax.plot(sub['Batch_Size'], sub['Throughput_samples_s'], label=label,
                        color=color, marker=marker, markersize=7, linewidth=2, linestyle=ls, alpha=0.95)

        if idx == 0:
            ax.legend(loc='upper left', frameon=True, facecolor=PANEL_COLOR, edgecolor='#444444')

    plt.tight_layout()
    out_path = 'plots/throughput_vs_batch_size.png'
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved {out_path}")

# ==============================================================================
# 3. PLOT 2: Training Comparison
# ==============================================================================
def plot_training_comparison():
    print("[3/6] Plotting training_comparison.png...")
    tr_cpu = pd.read_csv('data/training_cpu.csv')
    tr_gpu = pd.read_csv('data/training_gpu.csv')
    tr_tpu = pd.read_csv('data/training_tpu.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MobileViT-S CIFAR-10 Training Dynamics (CPU vs. GPU vs. TPU)', fontsize=16, weight='bold', y=0.98)

    epochs = [1, 2, 3]

    # Panel 1: Epoch Time (s)
    ax1 = axes[0, 0]
    ax1.set_title('(a) Total Epoch Time (seconds)', pad=10)
    ax1.plot(epochs, tr_cpu['Epoch_Time_s'], label='AMD EPYC CPU (FP32)', color=COLORS['CPU FP32'], marker='o', linewidth=2.5)
    ax1.plot(epochs, tr_gpu['Epoch_Time_s'], label='Tesla T4 GPU (FP32)', color=COLORS['GPU FP32'], marker='s', linewidth=2.5)
    ax1.plot(epochs, tr_tpu['Epoch_Time_s'], label='TPU v5e (FP32)', color=COLORS['TPU FP32'], marker='*', linewidth=2.5, markersize=9)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Time (s) [Log Scale]')
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax1.legend(facecolor=PANEL_COLOR, edgecolor='#444444')
    # Annotate TPU Epoch 1 compilation shock
    ax1.annotate('XLA Graph Compile Shock\n(53.88s total)', xy=(1, tr_tpu['Epoch_Time_s'][0]),
                 xytext=(1.3, 30), arrowprops=dict(facecolor=COLORS['TPU FP32'], shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, color=COLORS['TPU FP32'],
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_COLOR, edgecolor=COLORS['TPU FP32'], linewidth=1))

    # Panel 2: First Step vs Steady Step Latency
    ax2 = axes[0, 1]
    ax2.set_title('(b) Step Latency: First Step vs. Steady Median', pad=10)
    x = np.arange(3)
    width = 0.35
    first_steps = [tr_cpu['First_Step_ms'][0], tr_gpu['First_Step_ms'][0], tr_tpu['First_Step_ms'][0]]
    steady_steps = [tr_cpu['Steady_Median_Step_ms'][0], tr_gpu['Steady_Median_Step_ms'][0], tr_tpu['Steady_Median_Step_ms'][0]]
    
    b1 = ax2.bar(x - width/2, first_steps, width, label='First Step (ms)', color='#FF922B', alpha=0.85)
    b2 = ax2.bar(x + width/2, steady_steps, width, label='Steady Median Step (ms)', color='#339AF0', alpha=0.85)
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['AMD CPU', 'Tesla T4', 'TPU v5e'])
    ax2.set_ylabel('Latency (ms) [Log Scale]')
    ax2.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax2.legend(facecolor=PANEL_COLOR, edgecolor='#444444')
    # Annotate TPU 40.8s compile step with high-contrast box
    ax2.annotate('40,887 ms\n(XLA Compile)', xy=(2 - width/2, tr_tpu['First_Step_ms'][0]),
                 xytext=(1.45, 12000), arrowprops=dict(facecolor='#FFFFFF', shrink=0.05, width=1.5, headwidth=6),
                 fontsize=9.5, color='#FFFFFF', weight='bold',
                 bbox=dict(boxstyle='round,pad=0.35', facecolor=PANEL_COLOR, edgecolor='#FF922B', linewidth=1.2))

    # Panel 3: Training Loss Trajectory
    ax3 = axes[1, 0]
    ax3.set_title('(c) Cross-Entropy Loss Trajectory', pad=10)
    ax3.plot(epochs, tr_cpu['Train_Loss'], label='AMD CPU', color=COLORS['CPU FP32'], marker='o', linewidth=2)
    ax3.plot(epochs, tr_gpu['Train_Loss'], label='Tesla T4 GPU', color=COLORS['GPU FP32'], marker='s', linewidth=2)
    ax3.plot(epochs, tr_tpu['Train_Loss'], label='TPU v5e (FP32)', color=COLORS['TPU FP32'], marker='*', linewidth=2, markersize=9)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Loss')
    ax3.set_xticks(epochs)
    ax3.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax3.legend(facecolor=PANEL_COLOR, edgecolor='#444444')

    # Panel 4: Training Accuracy Trajectory
    ax4 = axes[1, 1]
    ax4.set_title('(d) Training Accuracy (%) Trajectory', pad=10)
    ax4.plot(epochs, tr_cpu['Train_Acc_%'], label='AMD CPU (89.55%)', color=COLORS['CPU FP32'], marker='o', linewidth=2)
    ax4.plot(epochs, tr_gpu['Train_Acc_%'], label='Tesla T4 GPU (89.94%)', color=COLORS['GPU FP32'], marker='s', linewidth=2)
    ax4.plot(epochs, tr_tpu['Train_Acc_%'], label='TPU v5e (89.89%)', color=COLORS['TPU FP32'], marker='*', linewidth=2, markersize=9)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Train Accuracy (%)')
    ax4.set_xticks(epochs)
    ax4.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax4.legend(facecolor=PANEL_COLOR, edgecolor='#444444')

    plt.tight_layout()
    out_path = 'plots/training_comparison.png'
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved {out_path}")

# ==============================================================================
# 4. PLOT 3: Resolution Scaling
# ==============================================================================
def plot_resolution_scaling():
    print("[4/6] Plotting resolution_scaling.png...")
    res_cg = pd.read_csv('data/resolution_sweep_cpu_gpu.csv')
    res_tpu = pd.read_csv('data/resolution_sweep_tpu.csv')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Input Resolution Scaling (BS=16): 128×128 vs. 256×256 vs. 384×384', fontsize=16, weight='bold', y=0.98)

    models = ['MobileNetV2', 'MobileViT-S']
    resolutions = [128, 256, 384]

    # Panel 1: MobileNetV2 (CNN - O(N) complexity)
    ax1 = axes[0]
    ax1.set_title('MobileNetV2 (Pure CNN)', pad=12)
    ax1.set_xlabel('Input Resolution (Edge Pixels)', labelpad=8)
    ax1.set_ylabel('Steady Median Latency (ms) [Log Scale]', labelpad=8)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)

    # Panel 2: MobileViT-S (Hybrid Vision Transformer - quadratic attention component)
    ax2 = axes[1]
    ax2.set_title('MobileViT-S (CNN + Transformer Hybrid)', pad=12)
    ax2.set_xlabel('Input Resolution (Edge Pixels)', labelpad=8)
    ax2.set_ylabel('Steady Median Latency (ms) [Log Scale]', labelpad=8)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)

    series_info = [
        ('CPU', 'FP32', res_cg, 'AMD CPU FP32', COLORS['CPU FP32'], 'o', '--'),
        ('GPU', 'FP32', res_cg, 'Tesla T4 FP32', COLORS['GPU FP32'], 's', '-'),
        ('GPU', 'FP16', res_cg, 'Tesla T4 FP16', COLORS['GPU FP16'], '^', '-'),
        ('TPU', 'BF16', res_tpu, 'TPU v5e BF16', COLORS['TPU BF16'], '*', '-')
    ]

    for dev, prec, df_source, label, color, marker, ls in series_info:
        for ax, m in zip([ax1, ax2], models):
            sub = df_source[(df_source['Device'] == dev) & (df_source['Precision'] == prec) & (df_source['Model'] == m)]
            if not sub.empty:
                sub = sub.sort_values('Resolution')
                ax.plot(sub['Resolution'], sub['Steady_Median_ms'], label=label,
                        color=color, marker=marker, markersize=8, linewidth=2, linestyle=ls)
                
                # Annotate values at 384x384 with non-overlapping offsets
                val_384 = sub[sub['Resolution'] == 384]['Steady_Median_ms'].values[0]
                
                # Custom vertical offsets to prevent label overlap
                if m == 'MobileNetV2':
                    if dev == 'GPU' and prec == 'FP16':  # 19.4ms
                        y_offset = 6
                    elif dev == 'TPU' and prec == 'BF16':  # 18.7ms
                        y_offset = -12
                    elif dev == 'GPU' and prec == 'FP32':  # 38.8ms
                        y_offset = -3
                    else:  # CPU (3617.1ms)
                        y_offset = -3
                else:  # MobileViT-S
                    if dev == 'TPU' and prec == 'BF16':  # 35.6ms
                        y_offset = -12
                    elif dev == 'GPU' and prec == 'FP16':  # 76.7ms
                        y_offset = 6
                    else:
                        y_offset = -3

                ax.annotate(f"{val_384:.1f}ms", (384, val_384), textcoords="offset points",
                            xytext=(8, y_offset), ha='left', va='center', fontsize=8.5, color=color, weight='bold')

    for ax in [ax1, ax2]:
        ax.set_xticks(resolutions)
        ax.set_xticklabels(['128×128', '256×256', '384×384'])
        ax.set_xlim(110, 430)  # Provide headroom for text labels
        ax.legend(loc='upper left', facecolor=PANEL_COLOR, edgecolor='#444444')

    plt.tight_layout()
    out_path = 'plots/resolution_scaling.png'
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved {out_path}")

# ==============================================================================
# 5. PLOT 4: CPU Profiler Breakdown
# ==============================================================================
def plot_cpu_profiler_breakdown():
    print("[5/6] Plotting cpu_profiler_breakdown.png...")
    df_ops = pd.read_csv('data/cpu_profiler_ops.csv')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
    fig.suptitle('PyTorch CPU Profiler: Top Kernel Execution Profiles (Self CPU %)', fontsize=16, weight='bold', y=0.98)

    models = ['MobileNetV2', 'MobileViT-S']
    colors_palette = ['#4DABF7', '#FF6B6B', '#FCC419', '#51CF66', '#CC5DE8', '#FF922B', '#20C997', '#845EF7', '#F06595', '#A9E34B']

    for idx, m in enumerate(models):
        ax = axes[idx]
        sub = df_ops[df_ops['Model'] == m].sort_values('Self_CPU_%', ascending=True)
        
        # Clean operator names for readability
        clean_names = [op.replace('aten::', '').replace('_scaled_dot_product_flash_attention_for_cpu', 'flash_attention')
                       for op in sub['Operator']]
        
        bars = ax.barh(clean_names, sub['Self_CPU_%'], color=colors_palette[:len(sub)], alpha=0.88, edgecolor='#333333')
        ax.set_title(f'{m} Top Operators', pad=12)
        ax.set_xlabel('Self CPU Time (%)', labelpad=8)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5, color=GRID_COLOR)
        
        # Add data labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.6, bar.get_y() + bar.get_height()/2.0, f"{width:.1f}%",
                    ha='left', va='center', fontsize=9, color=TEXT_COLOR, weight='bold')
        
        ax.set_xlim(0, max(sub['Self_CPU_%']) * 1.25)

    plt.tight_layout()
    out_path = 'plots/cpu_profiler_breakdown.png'
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved {out_path}")

# ==============================================================================
# 6. PLOT 5: Amdahl's Law Bottleneck
# ==============================================================================
def plot_amdahl_bottleneck():
    print("[6/6] Plotting amdahl_bottleneck.png...")
    df_amdahl = pd.read_csv('data/amdahl_analysis.csv')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.suptitle("Amdahl's Law System Bottlenecks: Serial Overhead vs. Theoretical Ceilings", fontsize=16, weight='bold', y=0.98)

    labels = [f"{r['Workload']}\n({r['Device']} {r['Precision']} {r['Batch_Size_or_Epoch']})"
              for _, r in df_amdahl.iterrows()]
    y_pos = np.arange(len(labels))

    # Panel 1: Stacked Bar of Compute vs Serial Overhead Fraction
    ax1 = axes[0]
    ax1.set_title('(a) Execution Time Partition: Compute vs. Serial Overhead', pad=32)
    compute_frac = 100.0 - df_amdahl['Serial_Fraction_%']
    serial_frac = df_amdahl['Serial_Fraction_%']

    ax1.barh(y_pos, compute_frac, label='Accelerable Compute Fraction (%)', color='#4DABF7', alpha=0.9)
    ax1.barh(y_pos, serial_frac, left=compute_frac, label='Serial / Interconnect Overhead (%)', color='#FF6B6B', alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Fraction of End-to-End Time (%)')
    ax1.set_xlim(0, 100)
    ax1.grid(True, axis='x', linestyle='--', alpha=0.5, color=GRID_COLOR)
    # Move legend above the axes to avoid obscuring any bars
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor=PANEL_COLOR, edgecolor='#444444')

    # Add text label of serial %
    for i, sf in enumerate(serial_frac):
        ax1.text(92, i, f"{sf:.1f}%", ha='center', va='center', fontsize=9, color='#FFFFFF', weight='bold')

    # Panel 2: Theoretical Speedup Ceiling (1 / f_serial) vs 2x Compute Speedup
    ax2 = axes[1]
    ax2.set_title("(b) Speedup Limit (s → ∞) vs. Realized Speedup (2× Faster Compute)", pad=32)
    bar_width = 0.38
    b1 = ax2.barh(y_pos - bar_width/2, df_amdahl['Speedup_Ceiling'], bar_width, label='Theoretical Ceiling (1 / f_serial)', color='#FCC419', alpha=0.9)
    b2 = ax2.barh(y_pos + bar_width/2, df_amdahl['Speedup_2x_Compute'], bar_width, label='Realized Speedup (2× Compute)', color='#51CF66', alpha=0.9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.set_xlabel('Speedup Factor (×)')
    ax2.set_xscale('log')
    ax2.set_xlim(1.0, 32.0)
    ax2.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax2.set_xticks([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0])
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5, color=GRID_COLOR)
    # Move legend above the axes to avoid covering the 18.70x MobileViT-S bar at the bottom!
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor=PANEL_COLOR, edgecolor='#444444')

    # Add text labels on bars
    for i, (ceil, s2) in enumerate(zip(df_amdahl['Speedup_Ceiling'], df_amdahl['Speedup_2x_Compute'])):
        ax2.text(ceil * 1.08, i - bar_width/2, f"{ceil:.2f}×", ha='left', va='center', fontsize=8.5, color='#FCC419', weight='bold')
        ax2.text(s2 * 1.08, i + bar_width/2, f"{s2:.2f}×", ha='left', va='center', fontsize=8.5, color='#51CF66', weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = 'plots/amdahl_bottleneck.png'
    plt.savefig(out_path, dpi=300, facecolor=BG_COLOR)
    plt.close()
    print(f"Saved {out_path}")

# ==============================================================================
# MAIN DRIVER
# ==============================================================================
if __name__ == '__main__':
    print("="*60)
    print("Running Day 9 Analysis & Plotting Pipeline")
    print("="*60)
    compute_amdahl_analysis()
    plot_throughput_vs_batch_size()
    plot_training_comparison()
    plot_resolution_scaling()
    plot_cpu_profiler_breakdown()
    plot_amdahl_bottleneck()
    print("="*60)
    print("All tasks completed successfully!")
    print("="*60)
