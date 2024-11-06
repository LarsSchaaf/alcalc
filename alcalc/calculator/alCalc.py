from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms
import datetime
from ase.io import read, write
import os
from mace.calculators.mace import MACECalculator
from IPython.core.debugger import Pdb
from ase.calculators.calculator import Calculator, all_changes

import argparse
import sys
import shutil
from numpy import random

import numpy as np
import logging
from glob import glob           #Wildcard searches unix style
from uuid import uuid4

import mace.cli.run_train



class AlMaceCalculator(MACECalculator):
    """
    AL_dir:               
    dft_calculator:  ASE calculator to run DFT with
    mlff_parameters: Parameters for MLFF (for example read_yaml("args-foundational.yaml") )
    al_threshold: Uncertainty/committee disagreement threshold to retrain model
    num_committes: Number of committee members in MLFF
    mlff_train_cmd: Command to train MLFF
    logger: Pre supplied logger
    initial: An already trained MLFF with correct number of neighbours                          
    initial_atom: Use this as an initial atom to start training the MLFF   TODO: allow user to input list of atoms                  
    num_rattle_configs: Number of times to rattle initial atom to generate an initial training set
    rattle_stdev: Standard deviation for rattling configs in ase (if using functionality)
    calculate_E0s: If True, calculates E0s for isolated atoms using level of DFT specified. Overrides any user given E0s. Set to False if you have already given E0s or want to use the MACE default method.
    e0s_vacuum: size of vacuum to place around isolated atoms when calcuating e0s. Set to None for no vacuum or cell
    """
    
    def __init__(
        self,
        AL_dir,
        dft_calculator,
        mlff_parameters,
        al_threshold,
        num_committes=4,
        mlff_train_cmd=None,
        logger=None,
        initial=None,                           # contains a already trained MLFF with correct number of neighbours
        initial_atom=None,                      # If no initial MLFF is given, use this atom to start training -> evaluate DFT + retrain
        num_rattle_configs=None,
        model_tag_s2='stagetwo',                # Tag put on model in mace after finishing stagetwo training. swa before mace 0.3.7
        rattle_stdev=0.005,
        calculate_E0s=True,                     # Calculate E0s for vacuum
        isolated_vacuum=30,                     # vacuum for calculating E0s around isolated atoms
        **kwargs,
    ):
        """Initialize the AL-MACE calculator with given parameters."""
        self.kwargs = kwargs

        #Transfer files from initial directory to AL_dir directory (working directory)
        #There was a clash here as couldn't make logger unless working directory exists, but couldn't log unless working directory exists
        if initial is not None and not os.path.exists(AL_dir):
            shutil.copytree(initial, AL_dir)

        #Add logger
        if logger is None:
            self.logger = logging.getLogger(__name__)
            # remove all handlers
            for handler in self.logger.handlers[
                :
            ]:  # Get a copy of the list to avoid modification issues
                self.logger.removeHandler(handler)
            
            if True:  # not self.logger.hasHandlers():
                print("Adding handlers")
                handler = logging.StreamHandler()
                # self.logger.addHandler(handler)
                # Create a FileHandler
                log_fname = os.path.join(AL_dir, f"logfile-{uuid4().hex}.log")
                handler = logging.FileHandler(log_fname)
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.info(f"Logging to {log_fname}")
                self.logger.info("Copied files from initial to current directory")
                self.logger.info("Initial: %s, AL_dir: %s", initial, AL_dir)
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger = logger
        
        self.logger.info("Initializing AL-MACE calculator: {}".format(AL_dir))

        

        # Name directories
        self.dir_AL = AL_dir
        self.dir_train = os.path.join(self.dir_AL, "TRAIN")
        self.dir_mlff = os.path.join(self.dir_AL, "MLFF")
        # os.chdir(self.dir_AL)

        # make all folders
        os.makedirs(self.dir_AL, exist_ok=True)
        os.makedirs(self.dir_train, exist_ok=True)
        [
            os.makedirs(os.path.join(self.dir_train, i), exist_ok=True)         #OS independant concatanation of path
            for i in ["new", "train"]
        ]
        os.makedirs(self.dir_mlff, exist_ok=True)
        self.dir_initial_working = os.getcwd()          #Current Working Directory (cwd)

        # other inputs
        self.dft_calculator = dft_calculator
        self.mlff_parameters = mlff_parameters
        self.mlff_train_cmd = mlff_train_cmd
        self.model_tag_s2 = model_tag_s2
        self.rattle_stdev = rattle_stdev
        self.num_rattle_configs = num_rattle_configs
        self.isolated_vacuum = isolated_vacuum

        # AL parameters
        self.al_threshold = al_threshold
        self.num_committes = num_committes

        #MACE E0s
        self.E0s = {}
        self.calculate_e0s = calculate_E0s

        # set to early time
        self.time_format = "%Y%m%d%H%M%S"
        self.timestamp_fail = datetime.datetime.min.strftime(self.time_format)
        self.timestamp_train = self.timestamp_fail
        print(self.timestamp_fail)

         


        #Run DFT on initial ase atom xyz (if present) and use to retrain MLFFs
        if initial_atom is not None:
            self.logger.info("Using initial atom")
            self.initial_atom = initial_atom.copy()
            self.logger.info("Initial atom: %s", self.initial_atom)
            self.calculate_dft(self.initial_atom)
            self.logger.info("Ran DFT on initial atom")
            self.initial_configs=[self.initial_atom]

            #Calculate E0s for isolated atoms
            if self.calculate_e0s is True:
                # Warn user if they have given E0s but set calculate_E0s that these will be overriden (TODO maybe opposite way better?)
                if 'E0s' in self.mlff_parameters:       
                    self.logger.info('!WARNING: calculate_E0s set to True but user has given E0s. Overriding user given E0s!')
                    self.logger.debug(f"User given E0s are: {mlff_parameters['E0s']}")
                self.get_E0s(self.initial_atom)
                
                self.mlff_parameters['E0s']=self.E0s
                self.logger.debug(f"Calculated E0s are: {mlff_parameters['E0s']}")
            
            #Rattle initial atom to get multiple initial configs - include a random seed
            if num_rattle_configs is not None:
                self.generate_rattled_configs()

            for config in self.initial_configs:
                self.create_new_training(config)
            self.timestamp_train = datetime.datetime.now().strftime(self.time_format)
            self.logger.info("Training Initial Force Field: %s", self.timestamp_train)
            self.update_mlffs()

       
        if len(self.get_fname_mlffs()) < self.num_committes and initial_atom is None:
            self.logger.info(
                "Not enough MLFFs to run committee: %d < %d",
                len(self.get_fname_mlffs()),
                self.num_committes,
            )
            self.logger.info(
                f"Make sure the initial model directory ({initial}) has sufficient amount of models."
                + "Or make sure you supply an initial traning set to initial_atom",
            )

        mace_fnames = self.update_mlffs()
        super().__init__(mace_fnames, **kwargs)


    def get_E0s(self, atoms):
        """Get E0s from base level of DFT"""
        self.logger.info("Calculating E0s")
        atnumbers=list(set(atoms.numbers))            #Atomic number
        atnumbers.sort()
        e0sfile=open('e0s.yaml', 'w')
        self.logger.info(f'Atomic numbers in initial atoms structure are: {atnumbers}')
        for atnum in atnumbers:
            atnum=int(atnum)
            isolated_atom = Atoms(numbers=[atnum])
            if self.isolated_vacuum is not None:
                isolated_atom.center(self.isolated_vacuum/2)
            self.calculate_dft(isolated_atom)
            self.E0s.update({atnum:isolated_atom.info["dft_energy"]})
            self.logger.info(f'{atnum}:\t {isolated_atom.get_chemical_formula()}\t {isolated_atom.info["dft_energy"]}')
        
        #Write e0s to yaml
        e0sfile.write(f"E0s: '{self.E0s}'")
        e0sfile.close()


    def generate_rattled_configs(self):
        self.logger.info(f'Creating {self.num_rattle_configs} rattled configs:')

        #Get a seed for rattling - user provided on random - for reproducability
        if 'random_rattle_seed' in self.kwargs:                                             #If user has provided seed
            self.grand_rattle_seed=self.kwargs['random_rattle_seed']
        else:                                                                               #Else generate own
            self.grand_rattle_seed=random.randint(1,99999999)
        random.seed(self.grand_rattle_seed)
        self.rattle_seeds=random.randint(1,99999999,size=self.num_rattle_configs)
        self.logger.info(f'Rattle seeds are {self.rattle_seeds}, generated using seed: {self.grand_rattle_seed}')

        #generate rattled configs with ase and run dft
        for rattle_seed in self.rattle_seeds:
            rattled_config=self.initial_atom.copy()                                         #Maybe make stdev user changeable
            rattled_config.rattle(stdev=self.rattle_stdev,seed=rattle_seed)
            self.calculate_dft(rattled_config)
            self.logger.info(f'Ran DFT on rattle seed {rattle_seed}')
            self.initial_configs.append(rattled_config)


    def get_fname_mlffs(self, current=True):
        """Get filenames of MLFF models."""
        
        #Look for swa or stagetwo models
        mlff_fname_pat = os.path.join(self.dir_mlff, str("{0}_{1}/*_"+self.model_tag_s2+".model"))
        mlff_fnames = np.array(sorted(glob(mlff_fname_pat.format("*", "*"))))
        if len(mlff_fnames) == 0:
            return []

        if current:
            timestamps = np.array(
                [int(fname.split("/")[-2].split("_")[1]) for fname in mlff_fnames]
            )

            mlff_fnames = mlff_fnames[timestamps >= int(self.timestamp_train)]

        return mlff_fnames

    def calculate(self, atoms=None, properties=["energy", "forces"], system_changes=all_changes):
        
        """Calculate energy and forces of the atoms."""
        super().calculate(atoms, properties, system_changes)
        self.mace.calculate(atoms)
        self.results = self.mace.results

        if "forces_comm" in self.results.keys():
            force_var = np.var(self.results["forces_comm"], axis=0)
        else:
            force_var = np.zeros_like(self.results["forces"])
        force_std = np.sqrt(np.sum(force_var, axis=1))
        self.logger.debug(f"Force std: {np.max(force_std)}")

        #If uncertainty measure exceeds that defined, retrain the model
        if np.max(force_std) > self.al_threshold:
            self.logger.info(
                f"AL threshold exceeded: {np.max(force_std)} > {self.al_threshold}"
            )
            self.timestamp_fail = datetime.datetime.now().strftime(self.time_format)
            new_at = self.calculate_dft(atoms)
            self.logger.info(f"Running DFT")
            self.create_new_training(new_at)
            self.timestamp_train = datetime.datetime.now().strftime(self.time_format)
            self.logger.info(f"Training Force Field: {self.timestamp_train}")
            self.update_mlffs()

    def calculate_dft(self, atoms):
        """Calculate energy and forces of atoms object using DFT calcultor"""
        self.dft_calculator.calculate(atoms)
        self.results = self.dft_calculator.results
        atoms.info["dft_energy"] = self.results["energy"]
        atoms.arrays["dft_forces"] = self.results["forces"]

        return atoms

    def create_new_training(self, atoms):
        """Save atomic configurations for further training."""
        
        #Save the atoms object to a new training xyz config
        filename = os.path.join(self.dir_train, f"new/new-{self.timestamp_fail}.xyz")
        write(filename, atoms, append=True)

        #Find and add all new training configs training config set
        all_new = glob(os.path.join(self.dir_train, "new/new-*.xyz"))
        traj_train = []
        [traj_train.extend(read(fname, index=":")) for fname in all_new]
        self.logger.info(f"Number of new training configs: {len(traj_train)}")
        filename = os.path.join(
            self.dir_train, f"train/train-{self.timestamp_fail}.xyz"
        )
        write(filename, traj_train)

        # for each committe create seperate train and valid split
        # TODO: cumulative new - for all new configurations created since failure
        # or since last training configuration that we start from.

        indices = np.arange(len(traj_train))
        # for i in number of committees
        for it in range(self.num_committes):
            np.random.seed(it)
            np.random.shuffle(indices)

            # Assuming a 90-10 split. Adjust if needed.
            split_at = int(0.9 * len(indices))
            train_idx, validation_idx = indices[:split_at], indices[split_at:]
            train = [traj_train[i] for i in train_idx]
            validation = [traj_train[i] for i in validation_idx]
            logging.info(
                f"Split {it+1} | Train indexes: {list(map(traj_train.index, train))} | "
                f"Validation indexes: {list(map(traj_train.index, validation))}"
            )
            write(filename.replace(".xyz", f"_{it}.xyz"), [atoms, *train])         
            write(filename.replace(".xyz", f"_{it}_val.xyz"), [atoms, *validation])
            self.current_train_fname = filename

    def save_train(self, atoms):
        """Save atomic configurations for further training."""
        filename = os.path.join(
            self.train_dir, "data", f"train/train-{self.timestamp_fail}.xyz"
        )
        write(filename, atoms, append=True)

    def update_mlffs(self):
        """Update MLFF models based on new training data."""
        for seed in range(self.num_committes):
            if len(self.get_fname_mlffs()) < self.num_committes:
                self.train_mace(
                    self.current_train_fname.replace(".xyz", f"_{seed}.xyz"),
                    self.current_train_fname.replace(".xyz", f"_{seed}_val.xyz"),
                    name=f"{self.timestamp_fail}_{self.timestamp_train}_{seed}",
                    seed=seed,
                )
            else:
                break
        assert (
            len(self.get_fname_mlffs()) >= self.num_committes
        ), f"Not enough MLFFs to run committee. Only {len(self.get_fname_mlffs())} and need {self.num_committes}"
        mace_fnames = sorted(self.get_fname_mlffs())[-self.num_committes :]
        self.logger.info("Using MLFFs: {}".format(mace_fnames))
        self.mace = MACECalculator(mace_fnames, **self.mlff_parameters)
        return mace_fnames

    def train_mace(self, train_fname, valid_fname, name, seed):
        # copy latest checkpoint
        fpat_checkpoint = os.path.join(
            self.dir_mlff, f"*_*/checkpoints/*{seed}_epoch-*_{self.model_tag_s2}.pt"
        )
        possible_checkpoints = glob(fpat_checkpoint)
        print(len(possible_checkpoints))
        print(fpat_checkpoint)
        if len(possible_checkpoints) > 0:
            # /home/lls34/rds/hpc-work/Data/Pynta/AL-calc/Dev/old-9-almost/MLFF/20231109123707_20231109123707/checkpoints/20231109123707_20231109123707_1_run-1_epoch-12_swa.pt

            sel_check = sorted(possible_checkpoints)[-1]            #Get latest checkpoint
            os.makedirs(
                os.path.join(
                    self.dir_mlff,
                    f"{self.timestamp_fail}_{self.timestamp_train}/checkpoints",
                ),
                exist_ok=True,
            )
            to_fname = os.path.join(
                self.dir_mlff,
                f"{self.timestamp_fail}_{self.timestamp_train}/checkpoints/{self.timestamp_fail}_{self.timestamp_train}_{seed}_run-{seed}_epoch-0.pt",
            )
            self.logger.info(f"Copying checkpoint: {sel_check} to {to_fname}")

            shutil.copy(sel_check, to_fname)
        # Create an argparse.Namespace object with the arguments
        args = argparse.Namespace(
            train_file=train_fname,
            valid_file=valid_fname,
            name=name,
            seed=seed,
            wandb_name=name,
            **self.mlff_parameters,
        )


        # Temporarily replace sys.argv (list of command and arguments)
        original_argv = sys.argv
        sys.argv = ["script_name"]  # replace 'script_name' with the actual script name
        for key, value in vars(args).items():
            if value is not None:
                if type(value) == bool and value is True:
                    sys.argv.extend([f"--{key}"])
                else:
                    sys.argv.extend([f"--{key}", str(value)])

        # Call the main function
        mlff_run_dir = os.path.join(
            self.dir_mlff, f"{self.timestamp_fail}_{self.timestamp_train}"
        )
        os.makedirs(mlff_run_dir, exist_ok=True)
        self.logger.info("Running MACE training in {}".format(mlff_run_dir))
        os.chdir(mlff_run_dir)
        try:
            # Create a new logger
            new_logger = logging.getLogger("mace")
            new_logger.handlers = []
            # Replace the root logger
            logging.root = new_logger
            # Call
            # sys.exit()
            mace.cli.run_train.main()
            # Restore the original root logger
            mace.cli.run_train.main()
        except Exception as e:
            os.chdir(self.dir_initial_working)
            raise e

        os.chdir(self.dir_initial_working)
        # Restore the original sys.argv
        sys.argv = original_argv

    def retrain_mlff(self):
        """Retrain the MLFF with new data."""
        train_configs = self.read_train_configs()
        run_dir = os.path.join(self.train_dir, "mlff")
        self.timestamp_train = datetime.datetime.now().strftime(self.time_format)
        mace_name = self.mace_name_pat.format(self.timestamp_train, self.timestamp_fail)
        ...

    def read_train_configs(self):
        """Read atomic configurations used for training."""
        data_dir = os.path.join(self.train_dir, "data")
        return [
            read(os.path.join(data_dir, fname))
            for fname in os.listdir(data_dir)
            if fname.endswith(".xyz")
        ]
