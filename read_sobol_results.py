import json
import numpy as np

def process_sobol_results(json_filename="sobol_raw_data.json", txt_output_name="sobol_printed_summary.txt"):
    """
    Reads Sobol sensitivity analysis results from a JSON file,
    prints a formatted text matrix to the terminal, and exports it to a file.
    """
    print(f"📖 Loading Sobol cluster results from {json_filename}...\n")
    with open(json_filename, "r") as json_file:
        sobol_data = json.load(json_file)
        
    # Open the text file to write out a mirrored version of our console printout
    with open(txt_output_name, "w") as out_file:
        
        # Loop through each metric calculated (e.g., annoyance_PA, loudness_sone)
        for metric_key, indices in sobol_data.items():
            display_title = metric_key.replace('_', ' ').title()
            
            names = indices['names']
            s1 = indices['S1']
            st = indices['ST']
            
            # Create a string buffer for the console layout
            lines = []
            lines.append("=" * 65)
            lines.append(f" Sobol Sensitivity Analysis: {display_title}")
            lines.append("=" * 65)
            lines.append(f"{'Variable Name':22s} | {'S1 (First-Order)':<16} | {'ST (Total-Order)':<16}")
            lines.append("-" * 65)
            
            # Zip and loop through parameters to append rows
            for name, s1_val, st_val in zip(names, s1, st):
                # Format to 4 decimal places
                lines.append(f"{name:22s} | {s1_val:<16.4f} | {st_val:<16.4f}")
            
            lines.append("-" * 65)
            
            # Check if Second-Order (pairwise) interactions were computed and stored
            if 'S2' in indices and indices['S2'] is not None:
                lines.append("\nSignificant Pairwise Interactions (S2 > 0.01):")
                lines.append("-" * 65)
                s2_matrix = np.array(indices['S2'])
                
                # S2 is a square matrix of size (num_vars x num_vars)
                num_vars = len(names)
                has_interactions = False
                
                for i in range(num_vars):
                    for j in range(i + 1, num_vars):
                        s2_val = s2_matrix[i, j]
                        # Only print relevant interactions to keep output clean
                        if not np.isnan(s2_val) and s2_val > 0.01:
                            lines.append(f" {names[i]} <--> {names[j]} : {s2_val:.4f}")
                            has_interactions = True
                            
                if not has_interactions:
                    lines.append(" No pairwise interactions exceeded the 0.01 threshold.")
                lines.append("-" * 65)
            
            lines.append("\n\n") # Spacing between metrics
            
            # Print everything out to console and write to text file simultaneously
            for line in lines:
                print(line)
                out_file.write(line + "\n")
                
    print(f"💾 Printed summary successfully exported to text file: {txt_output_name}")

# --- Run the Script ---
if __name__ == "__main__":
    # Ensure this matches your actual Sobol JSON file path
    process_sobol_results(json_filename="sobol_raw_data.json")

