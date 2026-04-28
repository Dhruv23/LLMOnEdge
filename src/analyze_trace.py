import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_trace_data(nvtx_file, gpu_file):
    print("Loading trace data...")
    nvtx_df = pd.read_csv(nvtx_file)
    gpu_df = pd.read_csv(gpu_file)

    # 1. Extract Phase Timings
    prefill_mask = nvtx_df['Name'].astype(str).str.contains('Prefill')
    decode_mask = nvtx_df['Name'].astype(str).str.contains('Decode')
    
    prefill_start = nvtx_df[prefill_mask]['Start (ns)'].values[0]
    prefill_end = nvtx_df[prefill_mask]['End (ns)'].values[0]

    decode_start = nvtx_df[decode_mask]['Start (ns)'].values[0]
    decode_end = nvtx_df[decode_mask]['End (ns)'].values[0]

    # 2. Extract Corunner executions
    corunner_df = gpu_df[gpu_df['Name'].astype(str).str.contains('cupy_sin__float32_float32')].copy()
    corunner_df = corunner_df.sort_values('Start (ns)')
    
    # 3. Calculate GPU-level Loop Latency
    corunner_df['Start (s)'] = corunner_df['Start (ns)'] / 1e9
    corunner_df['Latency (ms)'] = corunner_df['Start (ns)'].diff() / 1e6
    corunner_df = corunner_df.dropna(subset=['Latency (ms)'])

    # 4. Categorize by LLM Phase
    def assign_phase(row):
        t = row['Start (ns)']
        if prefill_start <= t <= prefill_end:
            return 'Prefill'
        elif decode_start <= t <= decode_end:
            return 'Decode'
        else:
            return 'Idle (No LLM)'

    corunner_df['Phase'] = corunner_df.apply(assign_phase, axis=1)

    # 5. Output Statistics
    stats = corunner_df.groupby('Phase')['Latency (ms)'].agg(['count', 'mean', 'median', 'max', 'min', 'std']).round(2)
    print("\n" + "="*50)
    print("--- GPU-Level Corunner Latency Stats by Phase (ms) ---")
    print("="*50)
    print(stats.to_string())
    print("="*50 + "\n")

    # --- PLOT 1: Full Timeline ---
    plt.figure(figsize=(12, 6))
    colors = {'Idle (No LLM)': 'grey', 'Prefill': 'red', 'Decode': 'blue'}
    plt.scatter(corunner_df['Start (s)'], corunner_df['Latency (ms)'], 
                c=corunner_df['Phase'].map(colors), s=4, alpha=0.8, label='Corunner Loop')
    plt.plot(corunner_df['Start (s)'], corunner_df['Latency (ms)'], color='black', alpha=0.3, linewidth=0.5)

    plt.axvspan(prefill_start/1e9, prefill_end/1e9, color='red', alpha=0.15, label='Prefill Window')
    plt.axvspan(decode_start/1e9, decode_end/1e9, color='blue', alpha=0.15, label='Decode Window')

    plt.title('Plot 1: Full Timeline - Corunner Latency Across LLM Phases')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Corunner Loop Latency (ms)')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot1_timeline.png', dpi=300)
    print("Plot 1 saved as 'plot1_timeline.png'")

    # --- PLOT 2: Boxplot Distribution ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Phase', y='Latency (ms)', data=corunner_df, palette=colors, order=['Idle (No LLM)', 'Prefill', 'Decode'])
    plt.title('Plot 2: Distribution of Corunner Latencies per Phase')
    plt.ylabel('Latency (ms)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot2_distribution_boxplot.png', dpi=300)
    print("Plot 2 saved as 'plot2_distribution_boxplot.png'")

    # --- PLOT 3: Histogram/KDE Distribution ---
    plt.figure(figsize=(10, 6))
    for phase in ['Idle (No LLM)', 'Prefill', 'Decode']:
        subset = corunner_df[corunner_df['Phase'] == phase]
        sns.kdeplot(subset['Latency (ms)'], label=phase, color=colors[phase], fill=True, alpha=0.3)
    
    plt.title('Plot 3: Density Distribution of Latencies')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot3_density.png', dpi=300)
    print("Plot 3 saved as 'plot3_density.png'")

    # --- PLOT 4: Zoom on Decode Phase Sawtooth ---
    plt.figure(figsize=(12, 5))
    decode_df = corunner_df[corunner_df['Phase'] == 'Decode']
    slice_df = decode_df.iloc[200:400] 
    
    plt.plot(slice_df['Start (s)'], slice_df['Latency (ms)'], color='blue', alpha=0.8, linewidth=1.5, marker='o', markersize=4)
    plt.title('Plot 4: Zoomed-in Decode Phase (The Sawtooth Jitter)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Corunner Loop Latency (ms)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot4_decode_zoom.png', dpi=300)
    print("Plot 4 saved as 'plot4_decode_zoom.png'")

    # --- PLOT 5: Bar Graph of Summary Statistics (WITH LABELS) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats_reordered = stats.reindex(['Idle (No LLM)', 'Prefill', 'Decode'])
    
    # Plot Mean, Median, and Max columns
    stats_reordered[['mean', 'median', 'max']].plot(
        kind='bar', 
        ax=ax, 
        color=['#4C72B0', '#55A868', '#C44E52'], 
        edgecolor='black',
        zorder=3
    )
    
    # ADD DATA LABELS HERE
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
    
    plt.title('Plot 5: Aggregate Interference (Mean, Median, Max Latency)')
    plt.ylabel('Corunner Loop Latency (ms)')
    plt.xlabel('LLM Phase')
    plt.xticks(rotation=0)
    
    # Extend the y-axis a bit to give room for labels
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

    plt.legend(['Mean Latency', 'Median Latency', 'Max Latency (Spike)'])
    plt.grid(axis='y', alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig('plot5_stats_bar.png', dpi=300)
    print("Plot 5 saved as 'plot5_stats_bar.png'")

if __name__ == "__main__":
    analyze_trace_data('gpt2_gating_profile_nvtx_pushpop_trace.csv', 'gpt2_gating_profile_cuda_gpu_trace.csv')