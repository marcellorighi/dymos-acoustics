import json
import numpy as np
import matplotlib.pyplot as plt

# 1. 🟢 paste your existing plotting function here exactly as it is:
def plot_morris_screening(morris_indices, metric_name="Acoustic Metric"):
    names = morris_indices['names']
    # Ensure they are numpy arrays for the max/linspace operations inside
    mu_star = np.array(morris_indices['mu_star'])
    sigma = np.array(morris_indices['sigma'])
    
    plt.figure(figsize=(8, 6), dpi=100)
    plt.scatter(mu_star, sigma, color='navy', s=100, zorder=3, edgecolors='black')
    
    for i, txt in enumerate(names):
        plt.annotate(txt, (mu_star[i], sigma[i]), textcoords="offset points", 
                     xytext=(8, 5), ha='left', fontsize=10, weight='bold')
    
    max_val = max(max(mu_star), max(sigma)) * 1.1
    x_line = np.linspace(0, max_val, 100)
    plt.plot(x_line, x_line, color='gray', linestyle='--', alpha=0.7, label=r'$\sigma = \mu^*$')
    
    plt.text(max_val * 0.1, max_val * 0.8, 'Highly Non-Linear /\nStrong Interactions', 
             color='darkred', fontsize=9, fontstyle='italic', alpha=0.7)
    plt.text(max_val * 0.6, max_val * 0.2, 'Linear /\nMonotonic Effects', 
             color='darkgreen', fontsize=9, fontstyle='italic', alpha=0.7)
    
    plt.title(f"Morris Sensitivity Screening: {metric_name}", fontsize=13, pad=15, weight='bold')
    plt.xlabel(r"Total Influence ($\mu^*$)", fontsize=11, labelpad=8)
    plt.ylabel(r"Non-Linearity & Interactions ($\sigma$)", fontsize=11, labelpad=8)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle=':', alpha=0.5, zorder=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    filename = f"local_morris_{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"🎨 Generated local plot: {filename}")
    plt.show()

# 2. 🟢 Read the JSON file and reconstruct the arrays
json_filename = "morris_raw_data.json"

print(f"📖 Loading cluster results from {json_filename}...")
with open(json_filename, "r") as json_file:
    raw_data = json.load(json_file)

# 3. 🟢 Loop over each acoustic metric stored in the file and plot it
for metric_key, metrics_dict in raw_data.items():
    # Format a clean display title based on the metric key
    display_title = metric_key.replace('_', ' ').title()
    
    # Pass the dictionary directly to your plotting engine
    plot_morris_screening(metrics_dict, metric_name=display_title)

print("✨ All plots generated successfully!")

