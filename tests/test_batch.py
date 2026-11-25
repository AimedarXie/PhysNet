"""Numerical checks using an untrained test network; no private checkpoint needed."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tensorflow.compat.v1 as tf
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from NNCalculator import NNCalculator
from neural_network.NeuralNetwork import NeuralNetwork


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.reset_default_graph()
        tf.set_random_seed(1234)
        np.random.seed(1234)
        cls.frames = [Atoms('OH2', positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
                      Atoms('OH2', positions=[[0, 0, 0], [1.06, 0.04, 0], [-0.3, 0.98, 0.1]])]
        def initialize(nn, sess, checkpoint):
            sess.run(tf.global_variables_initializer())
        with patch.object(NeuralNetwork, 'restore', initialize):
            cls.calc = NNCalculator(checkpoint='test-only', atoms=cls.frames[0],
                                    F=16, K=8, num_blocks=2, sr_cut=6.0,
                                    use_electrostatic=True, use_dispersion=True)

    @classmethod
    def tearDownClass(cls):
        cls.calc.sess.close()

    def sequential(self, frames):
        es, fs = [], []
        for atoms in frames:
            self.calc._calculate_all_properties(atoms)
            es.append(float(np.asarray(self.calc.last_energy).reshape(-1)[0]))
            fs.append(np.array(self.calc.last_forces, copy=True))
        return np.array(es), fs

    def assert_matches(self, frames):
        es, fs = self.sequential(frames)
        eb, fb = self.calc.get_energy_forces_batch(frames)
        np.testing.assert_allclose(np.asarray(eb).reshape(-1), es, atol=1e-5, rtol=1e-5)
        for a, b in zip(fb, fs):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)

    def test_single_and_multiple(self):
        self.assert_matches(self.frames[:1])
        self.assert_matches(self.frames)
        self.assert_matches(self.frames * 8)

    def test_mixed_atom_counts(self):
        self.assert_matches([self.frames[0], Atoms('H2', positions=[[0, 0, 0], [0.8, 0, 0]])])

    def test_independent_segments(self):
        e1, f1 = self.calc.get_energy_forces_batch(self.frames)
        changed = [a.copy() for a in self.frames]
        changed[1].positions[1, 0] += 0.3
        e2, f2 = self.calc.get_energy_forces_batch(changed)
        np.testing.assert_allclose(np.asarray(e1).reshape(-1)[0], np.asarray(e2).reshape(-1)[0], atol=1e-6)
        np.testing.assert_allclose(f1[0], f2[0], atol=1e-6)
        er, fr = self.calc.get_energy_forces_batch(self.frames[::-1])
        np.testing.assert_allclose(er, np.asarray(e1)[::-1], atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(fr, f1[::-1], atol=1e-5, rtol=1e-5)

    def test_energy_only_and_empty(self):
        e, _ = self.calc.get_energy_forces_batch(self.frames)
        np.testing.assert_allclose(self.calc.get_potential_energy_batch(self.frames), e)
        e, _ = self.calc.get_energy_forces_batch([])
        self.assertEqual(len(e), 0)
        with self.assertRaises(ValueError):
            self.calc.get_energy_forces_batch([Atoms()])


if __name__ == '__main__':
    unittest.main()
