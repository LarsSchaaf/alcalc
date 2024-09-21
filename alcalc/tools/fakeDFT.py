"""
Calculators used for testing AL calculator. 

"""

import os
from mace.calculators.mace import MACECalculator


class FakeDFTCalculator(MACECalculator):
    ''' A fast alternative to DFT calculations for testing purposes.'''
    def __init__(self, system, *args, **kwargs):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if system.lower() == "cu-catalyst":
            dft_mace = os.path.join(base_dir, "models", "CuCatalysis.model")
        else:
            raise ValueError(f"Unknown system: {system}")

        # Initialize the MACECalculator with the chosen model file and other args
        super().__init__(dft_mace, *args, **kwargs)
