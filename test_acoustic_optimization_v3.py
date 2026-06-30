import numpy as np
from pyoptsparse import Optimization, OPT

def calculate_annoyance(frequencies, magnitudes):
    """
    Calculates acoustic annoyance based on a simplified psychoacoustic model.
    Incorporates A-weighting approximation and roughness/sharpness heuristics.
    """
    # 1. A-weighting approximation (human ear sensitivity)
    # Most sensitive between 1kHz and 5kHz
    def a_weighting(f):
        f2 = f**2
        f4 = f**4
        f8 = f**8
        num = 1.258896e12 * f4
        den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.2**2)
        return 2.0 + 20 * np.log10(num / den)

    # 2. Loudness component (sum of weighted magnitudes)
    weighted_mags = []
    for f, m in zip(frequencies, magnitudes):
        w = a_weighting(f)
        weighted_mags.append(m + w)
    
    loudness_term = np.sum(np.power(10, np.array(weighted_mags) / 20))

    # 3. Sharpness component (penalizes high frequencies)
    # Higher frequencies contribute more to perceived 'sharpness'
    sharpness_term = np.sum(magnitudes * (frequencies / 1000.0) * 0.1)

    # 4. Roughness/Tonal component (simplified)
    # Penalizes tones that are close together but distinct (creates beating)
    roughness_term = 0
    if len(frequencies) > 1:
        sorted_f = np.sort(frequencies)
        diffs = np.diff(sorted_f)
        # Penalize differences in the 20-200Hz range (roughness band)
        roughness_term = np.sum(np.exp(-(diffs - 70)**2 / (2 * 30**2))) * 10

    total_annoyance = (0.5 * loudness_term) + (0.3 * sharpness_term) + (0.2 * roughness_term)
    return total_annoyance

def objective_function(xdict):
    """
    Optimization objective for pyoptsparse.
    xdict['freqs'] contains frequencies.
    xdict['mags'] contains magnitudes (dB).
    """
    freqs = xdict["freqs"]
    mags = xdict["mags"]
    
    annoyance = calculate_annoyance(freqs, mags)
    
    funcs = {}
    funcs["annoyance_obj"] = annoyance
    
    # Optional constraint: Total sound energy limit
    funcs["total_energy"] = np.sum(np.power(10, mags / 10))
    
    fail = False
    return funcs, fail

def run_acoustic_optimization():
    # Number of tones to optimize
    num_tones = 3
    
    opt_prob = Optimization("Acoustic_Annoyance_Minimization", objective_function)

    # Design Variables: Frequencies (Hz)
    # Range: 20Hz to 15,000Hz
    opt_prob.addVarGroup(
        "freqs", 
        nVars=num_tones, 
        varType="c", 
        value=np.array([440.0, 1000.0, 5000.0]), 
        lower=20.0, 
        upper=15000.0
    )

    # Design Variables: Magnitudes (dB)
    # Range: 30dB to 90dB
    opt_prob.addVarGroup(
        "mags", 
        nVars=num_tones, 
        varType="c", 
        value=np.array([60.0, 60.0, 60.0]), 
        lower=30.0, 
        upper=90.0
    )

    # Objective
    opt_prob.addObj("annoyance_obj")

    # Constraint: Total energy should not exceed a specific threshold (e.g., equivalent to 95dB)
    # This prevents the optimizer from simply setting all magnitudes to 30dB.
    opt_prob.addConGroup("total_energy", 1, lower=1e5, upper=1e10)

    # Solver Options for IPOPT
    opt_options = {
        "max_iter": 600,
        "tol": 1e-4,
        "print_level": 5,
        "nlp_scaling_method": "gradient-based",
        "limited_memory_update_type": "bfgs"
    }

    opt_solver = OPT("ipopt", options=opt_options)

    # Run optimization (using Finite Differences for the complex annoyance function)
    print("\n--- Optimizing Acoustic Profile via IPOPT ---")
    sol = opt_solver(opt_prob, sens="fd") 

    print("\n--- Optimized Acoustic Results ---")
    print(sol)
    
    # Extract results
    final_freqs = sol.xStar["freqs"]
    final_mags = sol.xStar["mags"]
    
    print(f"\nOptimal Frequencies (Hz): {final_freqs}")
    print(f"Optimal Magnitudes (dB): {final_mags}")

if __name__ == "__main__":
    run_acoustic_optimization()
