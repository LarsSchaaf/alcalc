# Define the base directory for the new location
# Works with mace: 1700caef7dcaaaea3f6d6635ffc603e0fd52de9c (Jul 16 11:35:44 2024)
import os
import shutil

from alcalc.calculator.alCalc import AlMaceCalculator
from mace.calculators.mace import MACECalculator
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from alcalc.tools.utils import read_yaml, read_yaml
from alcalc.tools.fakeDFT import FakeDFTCalculator


# If it terminates in the middle, you might be in the wrong directory, add line to change directory where file is
# os.chdir(FILE_DIR)

#VARIABLES
fname = "start.xyz"
direct = "/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/current"
src_dir = direct
base_new_dir = "/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/old-"

#MD
md_outfname="md3.xyz"
Nframes=100
write_interval=1
temperature_K = 300                 # For example, room temperature
friction_coefficient = 0.02         # 1/fs, this value depends on the system and desired damping
timestep = 1                        # md timestep in fs


# 1. Read in configuration to run MD on
at = read(fname)
args = read_yaml("args-foundational.yaml")


# 2. Define DFT calculator
# In this test case: take already trained model as DFT calculator
calc = FakeDFTCalculator(system="cu-catalyst", device="cuda", default_dtype="float32")




# 3. Change directory to a new directlry
# Just for testing
new_dir = base_new_dir + str(1)
while os.path.exists(new_dir):
    new_dir = base_new_dir + str(int(new_dir.split("-")[-1]) + 1)

# Move the entire directory tree to the new location
if False:  # os.path.exists(src_dir):
    shutil.move(src_dir, new_dir)
# move to directory

# alcalc = AlCalculator(model="model_type", config=ModelTypeConfig())

almace = AlMaceCalculator(
    AL_dir=direct,
    dft_calculator=calc,
    mlff_parameters=args,
    al_threshold=0.1,  # eV/A force error
    num_committes=2,
    initial="/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/fake-calculators+starting/Iteration4/initial",
    device="cuda",
    default_dtype="float32",
    initial_atom=at,
    num_rattle_configs=3
    # initial='/home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/initial-it10'
)

# 1. Can you run script
# 2. Can you use it in pynta with a single running process? (Use facke DFT calculator)
# 3. Does inttegration with pynta work for many simultanious runs? What are good thresholds 0.3 eV/A?
# 4. Can we reproduce the pytna run with ACTUAL dfT calculator.


almace.calculate(at)

print("Starting MD \n\n\n\n\n\n\n")




md = Langevin(at, timestep * units.fs, temperature_K * units.kB, friction_coefficient)


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


#MD

md = Langevin(at, 1 * units.fs, temperature_K * units.kB, friction_coefficient)
write(md_outfname, at)
md.attach(write, interval=1, filename=md_outfname, images=at, append=True)

def save_time(at, dyn):
    at.info["time"] = dyn.get_time() * units.fs

md.attach(save_time, interval=write_interval, at=at, dyn=md)
md.run(Nframes)


""" IF ABOVE NOT WORKING TRY BELOW MD
md = Langevin(at, 1 * units.fs, trajectory='md300.traj', temperature_K =temperature_K, friction=friction_coefficient)

def save_time(at, dyn):
    at.info["time"] = dyn.get_time() * units.fs

md.attach(save_time, interval=1, at=at, dyn=md)
md.run(50)
print('DONE')
"""

""" ToDo
typer: cli 
pip
testing
foundtion
parallel

"""
