import numpy as np
import openmdao.api as om

class AnnoyanceAnalyzer:
    """
    Enhanced logic for calculating Psychoacoustic Annoyance (PA) using 
    the four primary indicators: Loudness, Sharpness, Roughness, and Fluctuation Strength.
    """
    def calculate_annoyance(self, frequencies, magnitudes, duration=1.0, fs=44100):
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        waveform = np.zeros_like(t)
        for freq, mag in zip(frequencies, magnitudes):
            waveform += mag * np.sin(2 * np.pi * freq * t)
        
        # 1. Loudness (N) - Approximation of perceived volume
        rms = np.sqrt(np.mean(waveform**2)) + 1e-9
        N = 0.063 * (rms * 1000)**0.6
        
        # 2. Sharpness (S) - High-frequency weighting (Acum)
        # Weighting increases significantly above 3kHz
        freq_weights = 0.11 * (frequencies / 1000.0) * (0.1 + 0.9 * np.exp(0.0004 * frequencies))
        S = np.sum(magnitudes * freq_weights) / (np.sum(magnitudes) + 1e-9)
        
        # 3. Roughness (R) - High-frequency modulation (Asper)
        # Simplified: Roughness increases when tones are closely spaced (beating)
        R = 0
        if len(frequencies) > 1:
            for i in range(len(frequencies)):
                for j in range(i + 1, len(frequencies)):
                    df = abs(frequencies[i] - frequencies[j])
                    # Roughness peaks around 70Hz difference
                    if df < 300:
                        R += (magnitudes[i] * magnitudes[j]) * (df / 70.0) * np.exp(1 - df / 70.0)

        # 4. Fluctuation Strength (F) - Low-frequency modulation (Vacil)
        # Simplified: Lower frequency variations (up to 20Hz) cause 'wavering'
        # Here we model it based on very low frequency components or slow beats
        F = 0
        if len(frequencies) > 1:
            for i in range(len(frequencies)):
                for j in range(i + 1, len(frequencies)):
                    df = abs(frequencies[i] - frequencies[j])
                    if df < 20:
                        F += (magnitudes[i] * magnitudes[j]) * (df / 4.0) * np.exp(1 - df / 4.0)

        # Total Psychoacoustic Annoyance (PA) Formula (Simplified Fastl/Zwicker)
        # PA = N * (1 + sqrt(w_S^2 + w_FR^2))
        # where w_S is Sharpness contribution and w_FR is Fluctuation/Roughness contribution
        w_S = (S - 1.75) * 0.25 if S > 1.75 else 0
        w_FR = (2.18 / (N**0.4)) * (0.75 * F + 0.25 * R)
        
        annoyance = N * (1 + np.sqrt(w_S**2 + w_FR**2))
        return annoyance

class AnnoyanceComponent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_tones', default=4, types=int)
        self.analyzer = AnnoyanceAnalyzer()

    def setup(self):
        n = self.options['num_tones']
        self.add_input('freqs', val=np.linspace(500, 3000, n), units='Hz')
        self.add_input('mags', val=np.ones(n) * 0.2)
        self.add_output('total_annoyance', val=0.0)

    def compute(self, inputs, outputs):
        outputs['total_annoyance'] = self.analyzer.calculate_annoyance(inputs['freqs'], inputs['mags'])

def run_annoyance_optimization():
    prob = om.Problem()
    num_tones = 4
    
    prob.model.add_subsystem('annoyance_calc', 
                            AnnoyanceComponent(num_tones=num_tones), 
                            promotes=['*'])

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    
    # Add design variables with bounds
    prob.model.add_design_var('freqs', lower=50.0, upper=8000.0)
    prob.model.add_design_var('mags', lower=0.01, upper=1.0)
    
    # Minimize the aggregate annoyance
    prob.model.add_objective('total_annoyance')

    prob.setup()
    
    # Initial conditions
    prob.set_val('freqs', [800, 1200, 2500, 5000])
    prob.set_val('mags', [0.3, 0.3, 0.3, 0.3])

    prob.run_driver()

    print(f"Optimized Frequencies: {prob.get_val('freqs')}")
    print(f"Optimized Magnitudes: {prob.get_val('mags')}")
    print(f"Final Annoyance Score: {prob.get_val('total_annoyance')[0]}")

if __name__ == "__main__":
    run_annoyance_optimization()

