"""Predict all XYZ frames with one checkpoint; optionally compare sequential inference."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from NNCalculator import NNCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('xyz')
    parser.add_argument('--config', required=True, help='JSON with checkpoint and calculator options')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    atoms = read(args.xyz, index=':')
    if not atoms:
        parser.error('XYZ contains no configurations')
    with open(args.config) as f:
        options = json.load(f)
    calc = NNCalculator(atoms=atoms[0], **options)
    try:
        # Warm up both routes; benchmark excludes initialization/checkpoint loading.
        calc.get_energy_forces_batch(atoms)
        calc._calculate_all_properties(atoms[0])
        batch_times, sequential_times = [], []

        def batch():
            return calc.get_energy_forces_batch(atoms)

        def sequential():
            energies, forces = [], []
            for frame in atoms:
                # Force exactly one execution per frame, avoiding ASE cache effects.
                calc._calculate_all_properties(frame)
                energies.append(float(np.asarray(calc.last_energy).reshape(-1)[0]))
                forces.append(np.array(calc.last_forces, copy=True))
            return np.asarray(energies), forces

        for repeat in range(args.repeats):
            routes = ['batch', 'sequential'] if repeat % 2 == 0 else ['sequential', 'batch']
            for route in routes:
                if route == 'sequential' and not args.compare:
                    continue
                start = time.perf_counter()
                if route == 'batch':
                    energies, forces = batch()
                    batch_times.append(time.perf_counter() - start)
                else:
                    seq_e, seq_f = sequential()
                    sequential_times.append(time.perf_counter() - start)
        result = {'frames': len(atoms), 'energy': np.asarray(energies).reshape(-1).tolist(),
                  'forces': [np.asarray(f).tolist() for f in forces],
                  'units': 'model energy units; coordinates in Angstrom; forces = -dE/dR',
                  'batch_seconds': batch_times}
        if args.compare:
            de = float(np.max(np.abs(np.asarray(energies).reshape(-1) - seq_e)))
            df = max(float(np.max(np.abs(a-b))) for a, b in zip(forces, seq_f))
            result.update(energy_max_abs_error=de, force_max_abs_error=df,
                          sequential_seconds=sequential_times,
                          speedup=float(np.median(sequential_times)/np.median(batch_times)),
                          tolerance=args.atol, passed=bool(de <= args.atol and df <= args.atol))
        print(json.dumps(result, indent=2))
        if args.compare and not result['passed']:
            raise SystemExit(1)
    finally:
        calc.sess.close()


if __name__ == '__main__':
    main()
