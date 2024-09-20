
# Define the base directory for the new location
import os
import shutil

from alCalc import AlMaceCalculator
from mace.calculators.mace import MACECalculator
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import yaml

def write_yaml(file_path, dictionary):
    """Writes a given dictionary to a YAML file."""
    with open(file_path, 'w') as yaml_file:
        yaml.dump(dictionary, yaml_file, default_flow_style=False)

def read_yaml(file_path):
    """Reads a YAML file and returns the contents as a dictionary."""
    with open(file_path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)
    

# If it terminates in the middle, you might be in the wrong directory, add line to change directory where file is
# os.chdir(FILE_DIR)

# 1. Read in configuration to run MD on
fname = 'start.xyz'
at = read(fname)
args = read_yaml("args-foundational.yaml")


# 2. Define DFT calculator
# In this test case: take already trained model as DFT calculator
dft_mace = 'sa-Cu-it3-1e-4-1_swa.model'
calc = MACECalculator(
    dft_mace,
    device='cuda',
    default_dtype='float32',
)

direct = '/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/current'
src_dir = direct
base_new_dir = '/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/old-'


# 3. Change directory to a new directlry
# Just for testing
new_dir = base_new_dir + str(1)
while os.path.exists(new_dir):
    new_dir = base_new_dir + str(int(new_dir.split('-')[-1]) + 1)

# Move the entire directory tree to the new location
if False: #os.path.exists(src_dir):
    shutil.move(src_dir, new_dir)
# move to directory

# alcalc = AlCalculator(model="model_type", config=ModelTypeConfig())

almace = AlMaceCalculator(
    AL_dir=direct,
    dft_calculator=calc,
    mlff_parameters=args,
    al_threshold=0.1, # eV/A force error
    num_committes=2,
    initial='/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/fake-calculators+starting/Iteration4/initial',
    device='cuda', default_dtype='float32',
    initial_atom=at,
    # initial='/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/initial-it10'
)

# 1. Can you run script
# 2. Can you use it in pynta with a single running process? (Use facke DFT calculator)
# 3. Does inttegration with pynta work for many simultanious runs? What are good thresholds 0.3 eV/A?
# 4. Can we reproduce the pytna run with ACTUAL dfT calculator.


almace.calculate(at)

print('Starting MD \n\n\n\n\n\n\n')

temperature_K = 300  # For example, room temperature
friction_coefficient = 0.02  # 1/fs, this value depends on the system and desired damping


md = Langevin(at, 1 * units.fs, temperature_K * units.kB, friction_coefficient)


# Optionally initialize the velocities
MaxwellBoltzmannDistribution(at, temperature_K * units.kB)
print(at.get_forces())
at = at.copy()
# calc = MACECalculator(
#     dft_mace,
#     device='cuda',
#     default_dtype='float32',
# )
at.calc = almace
md = Langevin(at, 1 * units.fs, temperature_K * units.kB, friction_coefficient)
md.attach(write, interval=1, filename='md3.xyz', images=at, append=True)
def save_time(at, dyn):
    at.info['time'] = dyn.get_time() * units.fs
md.attach(save_time, interval=1, at=at, dyn=md)
md.run(10000)


''' ToDo
typer: cli 
pip
testing
foundtion
parallel

'''