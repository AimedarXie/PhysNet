import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import ase
from ase.neighborlist import neighbor_list
if __package__:
    from .neural_network.NeuralNetwork import *
    from .neural_network.activation_fn import *
else:
    from neural_network.NeuralNetwork import *
    from neural_network.activation_fn import *

'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class NNCalculator:
    #most parameters are just passed to the neural network
    def __init__(self,
                 checkpoint,                     #ckpt file from which to restore the model (can also be a list for ensembles)
                 atoms,                          #ASE atoms object
                 charge=0,                       #system charge
                 F=128,                          #dimensionality of feature vector
                 K=64,                           #number of radial basis functions
                 sr_cut=6.0,                     #short range cutoff distance 
                 lr_cut = None,                  #long range cutoff distance
                 num_blocks=5,                   #number of building blocks to be stacked
                 num_residual_atomic=2,          #number of residual layers for atomic refinements of feature vector
                 num_residual_interaction=3,     #number of residual layers for refinement of message vector
                 num_residual_output=1,          #number of residual layers for the output blocks
                 use_electrostatic=True,         #adds electrostatic contributions to atomic energy
                 use_dispersion=True,            #adds dispersion contributions to atomic energy
                 s6=None,                        #s6 coefficient for d3 dispersion, by default is learned
                 s8=None,                        #s8 coefficient for d3 dispersion, by default is learned
                 a1=None,                        #a1 coefficient for d3 dispersion, by default is learned
                 a2=None,                        #a2 coefficient for d3 dispersion, by default is learned   
                 activation_fn=shifted_softplus, #activation function
                 dtype=tf.float32):              #single or double precision

        self._np_dtype = dtype.as_numpy_dtype
        self._full_pair_cache = {}

        #create neighborlist
        if lr_cut is None:
            self._sr_cutoff = sr_cut
            self._lr_cutoff = None
            self._use_neighborlist = False
        else:
            self._sr_cutoff = sr_cut
            self._lr_cutoff = lr_cut
            self._use_neighborlist = True


        #save checkpoint
        self._checkpoint = checkpoint

        #create neural network
        self._nn = NeuralNetwork(F=F,
                                 K=K,
                                 sr_cut=sr_cut,
                                 lr_cut=lr_cut,
                                 num_blocks=num_blocks,                   
                                 num_residual_atomic=num_residual_atomic,          
                                 num_residual_interaction=num_residual_interaction,     
                                 num_residual_output=num_residual_output,          
                                 use_electrostatic=use_electrostatic,         
                                 use_dispersion=use_dispersion,      
                                 s6=s6,
                                 s8=s8,
                                 a1=a1,
                                 a2=a2,      
                                 activation_fn=activation_fn, 
                                 dtype=dtype, scope="neural_network")

        #create placeholders for feeding data
        self._Q_tot      = np.array(1*[charge])
        self._Z          = tf.placeholder(tf.int32, shape=[None, ], name="Z") 
        self._R          = tf.placeholder(dtype,    shape=[None,3], name="R")        
        self._idx_i      = tf.placeholder(tf.int32, shape=[None, ], name="idx_i") 
        self._idx_j      = tf.placeholder(tf.int32, shape=[None, ], name="idx_j") 
        self._offsets    = tf.placeholder(dtype,    shape=[None,3], name="offsets") 
        self._batch_seg  = tf.placeholder(tf.int32, shape=[None, ], name="batch_seg")
        self._sr_idx_i   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_i") 
        self._sr_idx_j   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_j") 
        self._sr_offsets = tf.placeholder(dtype,    shape=[None,3], name="sr_offsets") 
        
        #calculate atomic charges, energy and force evaluation nodes
        if self.use_neighborlist:
            Ea, Qa, Dij, nhloss = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets, self.sr_idx_i, self.sr_idx_j, self.sr_offsets)
        else:
            Ea, Qa, Dij, nhloss = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets)
        self._charges = self.nn.scaled_charges(self.Z, Qa, self.Q_tot, self.batch_seg)
        self._energy, self._forces = self.nn.energy_and_forces_from_scaled_atomic_properties(Ea, self.charges, Dij, self.Z, self.R, self.idx_i, self.idx_j, self.batch_seg)

        ##create TensorFlow session and load neural network(s)
        self._sess = tf.Session()
        #config = tf.ConfigProto()
        #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        #self._sess = tf.Session(config=config)
        if(type(self.checkpoint) is not list):
            self.nn.restore(self.sess, self.checkpoint)

        #calculate properties once to initialize everything
        self._calculate_all_properties(atoms)

    def calculation_required(self, atoms, quantities=None):
        return atoms != self.last_atoms

    def _calculate_all_properties(self, atoms):
        #find neighbors and offsets
        if self.use_neighborlist or any(atoms.get_pbc()):
            idx_i, idx_j, S = neighbor_list('ijS', atoms, self.lr_cutoff)
            offsets = np.dot(S, atoms.get_cell())
            sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, self.sr_cutoff)
            sr_offsets = np.dot(sr_S, atoms.get_cell())
            batch_seg = np.zeros(len(atoms), dtype=np.int32)
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets,
                    self.sr_idx_i: sr_idx_i, self.sr_idx_j: sr_idx_j, self.sr_offsets: sr_offsets,
                    self.batch_seg: batch_seg}
        else:
            N = len(atoms)
            cached = self._full_pair_cache.get(N)
            if cached is None:
                all_idx = np.arange(N, dtype=np.int32)
                idx_i = np.repeat(all_idx, N)
                idx_j = np.tile(all_idx, N)
                mask = idx_i != idx_j
                idx_i = idx_i[mask]
                idx_j = idx_j[mask]
                offsets = np.zeros((idx_i.shape[0], 3), dtype=self._np_dtype)
                self._full_pair_cache[N] = (idx_i, idx_j, offsets)
            else:
                idx_i, idx_j, offsets = cached
            batch_seg = np.zeros(N, dtype=np.int32)
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets,
                    self.batch_seg: batch_seg}

        #calculate energy and forces (in case multiple NNs are used as ensemble, this forms the average)
        if(type(self.checkpoint) is not list): #only one NN
            self._last_energy, self._last_forces, self._last_charges = self.sess.run([self.energy, self.forces, self.charges], feed_dict=feed_dict)
            self._energy_stdev = 0
        else: #ensemble is used
            for i in range(len(self.checkpoint)):
                self.nn.restore(self.sess, self.checkpoint[i])
                energy, forces, charges = self.sess.run([self.energy, self.forces, self.charges], feed_dict=feed_dict)
                if i == 0:
                    self._last_energy  = energy
                    self._last_forces  = forces
                    self._last_charges = charges
                    self._energy_stdev = 0
                else:
                    n = i+1
                    delta = energy-self.last_energy
                    self._last_energy += delta/n
                    self._energy_stdev += delta*(energy-self.last_energy)
                    self._last_charges += (charges-self.last_charges)/n
                    self._last_forces += (forces-self.last_forces)/n
            if(len(self.checkpoint) > 1):
                self._energy_stdev = np.sqrt(self.energy_stdev/len(self.checkpoint))

        self._last_energy = np.array(1*[self.last_energy]) #prevents some problems...

        #store copy of atoms
        self._last_atoms = atoms.copy()

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces

    def get_charges(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_charges

    def get_potential_energy_batch(self, atoms_list):
        energies, _ = self.get_energy_forces_batch(atoms_list, need_forces=False)
        return energies

    def get_energy_forces_batch(self, atoms_list, need_forces=True):
        """Evaluate independent configurations in one session call (one checkpoint)."""
        if isinstance(self.checkpoint, list):
            raise NotImplementedError("Batch evaluation requires a single checkpoint")
        if any(len(atoms) == 0 for atoms in atoms_list):
            raise ValueError("Empty configurations are not supported")
        if len(atoms_list) == 0:
            return np.array([], dtype=self._np_dtype), np.array([], dtype=self._np_dtype)

        first_pbc = atoms_list[0].get_pbc()
        if any((atoms.get_pbc() != first_pbc).any() for atoms in atoms_list):
            raise ValueError("All atoms must have the same PBC settings for batch evaluation.")

        if any(first_pbc) and self.lr_cutoff is None:
            raise ValueError("Periodic batch evaluation requires lr_cut")
        use_nl = self.use_neighborlist or any(first_pbc)
        Z_all = []
        R_all = []
        batch_seg = []
        idx_i_all = []
        idx_j_all = []
        offsets_all = []
        sr_idx_i_all = []
        sr_idx_j_all = []
        sr_offsets_all = []
        counts = []
        base = 0

        for batch_idx, atoms in enumerate(atoms_list):
            Z = atoms.get_atomic_numbers()
            R = atoms.get_positions()
            n_atoms = len(atoms)
            counts.append(n_atoms)
            Z_all.append(Z)
            R_all.append(R)
            batch_seg.append(np.full(n_atoms, batch_idx, dtype=np.int32))

            if use_nl:
                idx_i, idx_j, S = neighbor_list('ijS', atoms, self.lr_cutoff)
                offsets = np.dot(S, atoms.get_cell())
                sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, self.sr_cutoff)
                sr_offsets = np.dot(sr_S, atoms.get_cell())
            else:
                cached = self._full_pair_cache.get(n_atoms)
                if cached is None:
                    all_idx = np.arange(n_atoms, dtype=np.int32)
                    idx_i = np.repeat(all_idx, n_atoms)
                    idx_j = np.tile(all_idx, n_atoms)
                    mask = idx_i != idx_j
                    idx_i = idx_i[mask]
                    idx_j = idx_j[mask]
                    offsets = np.zeros((idx_i.shape[0], 3), dtype=self._np_dtype)
                    self._full_pair_cache[n_atoms] = (idx_i, idx_j, offsets)
                else:
                    idx_i, idx_j, offsets = cached

            idx_i_all.append(idx_i + base)
            idx_j_all.append(idx_j + base)
            offsets_all.append(offsets)
            if use_nl:
                sr_idx_i_all.append(sr_idx_i + base)
                sr_idx_j_all.append(sr_idx_j + base)
                sr_offsets_all.append(sr_offsets)
            base += n_atoms

        feed_dict = {
            self.Z: np.concatenate(Z_all),
            self.R: np.concatenate(R_all),
            self.idx_i: np.concatenate(idx_i_all),
            self.idx_j: np.concatenate(idx_j_all),
            self.offsets: np.concatenate(offsets_all),
            self.batch_seg: np.concatenate(batch_seg),
        }
        if use_nl:
            feed_dict[self.sr_idx_i] = np.concatenate(sr_idx_i_all)
            feed_dict[self.sr_idx_j] = np.concatenate(sr_idx_j_all)
            feed_dict[self.sr_offsets] = np.concatenate(sr_offsets_all)

        if need_forces:
            energies, forces = self.sess.run([self.energy, self.forces], feed_dict=feed_dict)
        else:
            energies = self.sess.run(self.energy, feed_dict=feed_dict)
            forces = None

        if forces is None:
            return energies, None

        if len(set(counts)) == 1:
            n_atoms = counts[0]
            forces = forces.reshape((len(atoms_list), n_atoms, 3))
        else:
            split_idx = np.cumsum(counts)[:-1]
            forces = np.split(forces, split_idx)
        return energies, forces

    @property
    def sess(self):
        return self._sess

    @property
    def last_atoms(self):
        return self._last_atoms

    @property
    def last_energy(self):
        return self._last_energy

    @property
    def last_forces(self):
        return self._last_forces

    @property
    def last_charges(self):
        return self._last_charges

    @property
    def energy_stdev(self):
        return self._energy_stdev

    @property
    def sr_cutoff(self):
        return self._sr_cutoff

    @property
    def lr_cutoff(self):
        return self._lr_cutoff

    @property
    def use_neighborlist(self):
        return self._use_neighborlist

    @property
    def nn(self):
        return self._nn

    @property
    def checkpoint(self):
        return self._checkpoint

    @property
    def Z(self):
        return self._Z

    @property
    def Q_tot(self):
        return self._Q_tot

    @property
    def R(self):
        return self._R

    @property
    def offsets(self):
        return self._offsets

    @property
    def batch_seg(self):
        return self._batch_seg

    @property
    def idx_i(self):
        return self._idx_i

    @property
    def idx_j(self):
        return self._idx_j

    @property
    def sr_offsets(self):
        return self._sr_offsets

    @property
    def sr_idx_i(self):
        return self._sr_idx_i

    @property
    def sr_idx_j(self):
        return self._sr_idx_j

    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces

    @property
    def charges(self):
        return self._charges
