import numpy as np
import openmdao.api as om

class AnnoyanceAnalyzer:
    """
    Core logic for calculating psychoacoustic annoyance indicators.
    This uses a simplified model based on Zwicker's parameters (Loudness and Sharpness).
    """
    def calculate_annoyance(self, frequencies, magnitudes, duration=1.0, fs=44100):
        # Generate time vector
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.zeros_like(t)
        
        # Generate synthetic signal from optimization design variables
        for freq, mag in zip(frequencies, magnitudes):
            waveform += mag * np.sin(2 * np.pi * freq * t)
        
        # 1. Calculate Loudness (N) approximation
        # Based on RMS of the signal shifted to a power-law scale
        rms = np.sqrt(np.mean(waveform**2)) + 1e-9
        loudness = 0.063 * (rms * 1000)**0.6
        
        # 2. Calculate Sharpness (S) approximation
        # Higher frequencies are weighted more heavily (Acum scale approximation)
        # We weight the input magnitudes by their frequency positions
        freq_weights = (frequencies / 3000.0)**1.2
        total_mag = np.sum(magnitudes) + 1e-9
        sharpness = np.sum(magnitudes * freq_weights) / total_mag
        
        # 3. Calculate Psychoacoustic Annoyance (A)
        # A standard simplified model: A = N * (1 + 0.25 * S)
        annoyance = loudness * (1 + 0.25 * sharpness)
        
        return annoyance

class AnnoyanceComponent(om.ExplicitComponent):
    """
    OpenMDAO Component wrapper for the Psychoacoustic Annoyance logic.
    """
    def initialize(self):
        # Define the number of tones we want to optimize
        self.options.declare('num_tones', default=3, types=int)
        self.analyzer = AnnoyanceAnalyzer()

    def setup(self):
        n = self.options['num_tones']
        
        # Inputs: Frequencies and Magnitudes for each tone
        self.add_input('freqs', val=np.linspace(200, 4000, n), units='Hz')
        self.add_input('mags', val=np.ones(n) * 0.1)
        
        # Output: The computed annoyance value (The Objective)
        self.add_output('total_annoyance', val=0.0)

    def compute(self, inputs, outputs):
        freqs = inputs['freqs']
        mags = inputs['mags']
        
        # Call the external logic
        outputs['total_annoyance'] = self.analyzer.calculate_annoyance(freqs, mags)

def run_annoyance_optimization():
    # Configuration
    num_tones = 4
    prob = om.Problem()
    model = prob.model

    # Add the annoyance subsystem
    model.add_subsystem('annoyance_system', 
                        AnnoyanceComponent(num_tones=num_tones), 
                        promotes=['*'])

    # Setup Optimizer: Using COBYLA (Gradient-free) 
    # as psychoacoustic models can have complex topologies
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'COBYLA'
    prob.driver.options['maxiter'] = 300
    prob.driver.options['tol'] = 1e-4

    # Design Variable 1: Frequencies
    # Range limited to typical sensitive human hearing range
    prob.model.add_design_var('freqs', lower=100.0, upper=6000.0)
    
    # Design Variable 2: Magnitudes
    # Constraints ensure the signal doesn't just vanish
    prob.model.add_design_var('mags', lower=0.05, upper=1.0)

    # Objective: Minimize total annoyance
    prob.model.add_objective('total_annoyance')

    # Setup and Initial Values
    prob.setup()
    
    # Start with a "harsh" initial guess (high frequencies, high magnitudes)
    prob.set_val('freqs', np.array([1000, 2500, 4000, 5500]))
    prob.set_val('mags', np.array([0.5, 0.5, 0.5, 0.5]))

    print("--- Executing OpenMDAO Optimization ---")
    prob.run_driver()
    print("--- Optimization Complete ---\n")

    # Output Results
    opt_freqs = prob.get_val('freqs')
    opt_mags = prob.get_val('mags')
    opt_annoyance = prob.get_val('total_annoyance')

    print(f"Optimal Frequencies (Hz): {np.round(opt_freqs, 2)}")
    print(f"Optimal Magnitudes:       {np.round(opt_mags, 3)}")
    print(f"Resulting Annoyance:      {opt_annoyance[0]:.6f}")

if __name__ == "__main__":
    run_annoyance_optimization()

