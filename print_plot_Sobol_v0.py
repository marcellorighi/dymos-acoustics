import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from SALib.analyze import sobol as sobol_analyzer

# 1. Re-declare the IDENTICAL Problem Metadata used in your simulation script
problem = {
    'num_vars': 6,
    'names': ['Kp_Velocity', 'Ki_Velocity', 'Kp_Attitude', 'Kd_Attitude', 'Kp_Altitude', 'Kd_Altitude'],
    'bounds': [[0.05, 0.25], [0.01, 0.08], [2.0, 4.0], [0.3, 0.8], [10.0, 20.0], [4.0, 10.0]]
}

# Mapping the CSV columns to clean display titles for your plots
metrics_to_analyze = {
    "dose_annoyance_PA": "Psychoacoustic Annoyance",
    "dose_loudness_sone": "Loudness",
    "dose_sharpness_acum": "Sharpness",
    "dose_roughness_asper": "Roughness",
    "dose_fluctuation_vacil": "Fluctuation Strength"
}

# 2. Load the dataset from disk
csv_filename = "sobol_simulation_results.csv"
print(f"Loading pre-computed simulation matrix from '{csv_filename}'...")
df = pd.read_csv(csv_filename)

# Set global plotting style parameters for clean academic figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10})

# 3. Perform the Sobol Decompositions and Plotting
for column_key, display_name in metrics_to_analyze.items():
    print(f"\n==========================================")
    print(f" Sobol Sensitivity Analysis: {display_name}")
    print(f"==========================================")
    
    # Pull the specific target metric vector out of the dataset
    Y = df[column_key].to_numpy()
    
    # Run the SALib analyzer math instantly
    Si = sobol_analyzer.analyze(problem, Y, calc_second_order=True)
    
    # --- PRINT TEXT RESULTS ---
    print(f"\n--- MAIN EFFECTS (S1 & ST) ---")
    print(f"{'Gain Parameter':20s} | {'S1 ± Conf':<16} | {'ST ± Conf':<16}")
    print("-" * 60)
    for idx, name in enumerate(problem['names']):
        print(f"{name:20s} | {Si['S1'][idx]:.3f} ± {Si['S1_conf'][idx]:.3f} | {Si['ST'][idx]:.3f} ± {Si['ST_conf'][idx]:.3f}")
        
    print(f"\n--- SECOND-ORDER INTERACTIONS (S2 Matrix Pairs) ---")
    print(f"{'Parameter Pair':42s} | {'S2 ± Conf':<16}")
    print("-" * 65)
    for idx1 in range(problem['num_vars']):
        for idx2 in range(idx1 + 1, problem['num_vars']):
            p_pair = f"{problem['names'][idx1]} <-> {problem['names'][idx2]}"
            s2_val = Si['S2'][idx1, idx2]
            s2_conf = Si['S2_conf'][idx1, idx2]
            s2_str = "NaN (Low Samples)" if np.isnan(s2_val) else f"{s2_val:.3f} ± {s2_conf:.3f}"
            print(f"{p_pair:42s} | {s2_str:<16}")

    # --- PLOT 1: S1 VS ST SENSITIVITY BAR CHART WITH CONFIDENCE INTERVALS ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x_indices = np.arange(problem['num_vars'])
    bar_width = 0.35
    
    # Clean up any negative sampling noise values to exactly 0 for visual clarity
    s1_clean = np.clip(Si['S1'], 0, 1)
    st_clean = np.clip(Si['ST'], 0, 1)
    
    # Plot S1 Bars
    ax.bar(x_indices - bar_width/2, s1_clean, bar_width, 
           yerr=Si['S1_conf'], label='First-Order ($S_1$)', 
           color='#3498db', edgecolor='black', capsize=4, alpha=0.9)
    
    # Plot ST Bars
    ax.bar(x_indices + bar_width/2, st_clean, bar_width, 
           yerr=Si['ST_conf'], label='Total-Order ($S_T$)', 
           color='#e74c3c', edgecolor='black', capsize=4, alpha=0.9)
    
    ax.set_ylabel('Sensitivity Index Fraction')
    ax.set_title(f'Sobol Main Effects Sensitivity: {display_name}', fontweight='bold', pad=15)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(problem['names'], rotation=25, ha='right')
    ax.set_ylim(0, max(max(st_clean + Si['ST_conf']), 1.05)) # Scale dynamically but safely
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    plt.tight_layout()
    plot1_filename = f"sobol_main_effects_{column_key}.png"
    plt.savefig(plot1_filename, dpi=300)
    plt.close()
    print(f"-> Saved main effects bar chart to '{plot1_filename}'")

    # --- PLOT 2: SECOND-ORDER (S2) INTERACTION MATRIX HEATMAP ---
    # Construct a clean symmetric matrix out of the flat combinations
    num_vars = problem['num_vars']
    s2_matrix = np.zeros((num_vars, num_vars))
    
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                s2_matrix[i, j] = 0.0  # Self-interaction is zero by definition
            elif i < j:
                # Fill upper triangle using clean values (clipping negative sampling artifacts)
                val = Si['S2'][i, j]
                s2_matrix[i, j] = max(0, val) if not np.isnan(val) else 0.0
            else:
                # Mirror it onto lower triangle for symmetry
                val = Si['S2'][j, i]
                s2_matrix[i, j] = max(0, val) if not np.isnan(val) else 0.0

    # Mask the upper triangle out if you prefer a clean diagonal display, or leave wide open
    mask = np.triu(np.ones_like(s2_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(s2_matrix, mask=mask, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=problem['names'], yticklabels=problem['names'],
                linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Interaction Index Value ($S_2$)'}, ax=ax)
    
    ax.set_title(f'Second-Order Interactions ($S_2$): {display_name}', fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plot2_filename = f"sobol_interactions_{column_key}.png"
    plt.savefig(plot2_filename, dpi=300)
    plt.close()
    print(f"-> Saved interaction matrix heatmap to '{plot2_filename}'")

print("\nAll analyses complete! Check your folder for the generated text summaries and high-resolution .png figures.")