# PhysNet

Tensorflow implementation of PhysNet (see https://arxiv.org/abs/1902.08408) for details

## Batch inference extension / 批量推理扩展

This branch adds segment-batched energy and force inference to the original
PhysNet calculator. It is adapted from an RPMDrate deployment for the
O + dimethyl ether (DME) system. The original MIT license and citation are retained.

本分支将多个独立构型（例如 RPMD 的多个珠子）合并到一次模型推理中。
每个构型分别生成原子对，使用 `batch_seg` 分段汇总电荷和能量，不引入跨构型的模型相互作用。
返回每个构型的能量和力，不做珠子平均。核心网络结构和已加载的模型参数不因 batch 改动而改变。

### Quick start

Use a Python environment with NumPy, ASE and TensorFlow. The inference code uses
`tensorflow.compat.v1` graph mode. TensorFlow 2.5.0 / ASE 3.22.1 is the source
deployment; this is not a claim of compatibility with all newer TensorFlow releases.
The original training code below is retained and has not been ported or revalidated
for TensorFlow 2. Importing this calculator disables TF2 eager behavior.

```python
from NNCalculator import NNCalculator

calc = NNCalculator(checkpoint="/path/to/model.ckpt", atoms=atoms_list[0],
                    F=64, K=32, num_blocks=3)  # match ALL model hyperparameters
try:
    energies, forces = calc.get_energy_forces_batch(atoms_list)
finally:
    calc.sess.close()
```

For equal atom counts, forces have shape `(B, N, 3)`; for unequal counts a list
of `(N_i, 3)` arrays is returned. Coordinates are in Angstrom. Energy units follow
the trained model; force units are energy/Angstrom. Do not assume every checkpoint
uses kcal/mol. Forces are **minus** the energy gradient. A singleton energy may be
a scalar because of the upstream squeeze operation; normalize with
`np.asarray(energies).reshape(-1)` when consuming it.

### XYZ prediction and comparison

Copy `examples/batch_inference/config.example.json` to `config.local.json`, then
set your checkpoint path and its exact architecture options. Weights are not bundled.

```bash
python examples/batch_inference/predict.py your_frames.xyz --config config.local.json
python examples/batch_inference/predict.py your_frames.xyz --config config.local.json --compare --repeats 5 > benchmark.local.json
```

For reproducible FP32 comparisons on NVIDIA Ampere GPUs, set
`export NVIDIA_TF32_OVERRIDE=0` **before starting Python**. TF32 may select different
matrix kernels for sequential and batched shapes and noticeably change forces.
Always compare both energy and forces with your actual checkpoint and precision settings.

The supplied water XYZ demonstrates the file format only; use configurations
within your model's training domain. The comparison performs exactly one session
execution per configuration in the sequential route, warms up both routes, alternates
their timing order, and excludes loading/initialization. It reports maximum energy
and force differences, timing samples and median speedup. The default absolute
tolerance is `1e-4` in model units; select `--atol` appropriate to model precision.
Calling batch once reduces session-call overhead; it does not guarantee a speedup
equal to the number of configurations. Large batches require more memory.

Run numerical tests without any model weights:

```bash
python -m unittest discover -s tests -v
```

Tests use a small, randomly initialized neural network, with electrostatics and
dispersion enabled. They check sequential/batch equivalence for 1, 2 and 16 frames,
mixed atom counts, permutation, segment independence and energy-only evaluation.
They validate the batching implementation, not the physical quality of a PES.

Recorded validation (2026-11-25): four CPU numerical tests passed. On an RTX 3090,
TF 2.5.0 with TF32 disabled, 16 slightly perturbed ODME configurations and three
timing repeats gave:

| Model | Sequential median | Batch median | Speedup | Max energy difference (kcal/mol) | Max force difference (kcal/mol/Angstrom) |
|---|---:|---:|---:|---:|---:|
| DME | 0.15365 s | 0.01077 s | 14.27 | 1.91e-6 | 9.06e-6 |
| diff | 0.21644 s | 0.01422 s | 15.22 | 3.58e-7 | 1.31e-6 |

Raw timing samples and test metadata are in `tests/validation_summary.json`.
These results use external private checkpoints and a small synthetic configuration
set; they are not an end-to-end RPMD benchmark or a general accuracy guarantee.

### RPMDrate example / 双势能接口

`examples/rpmd_interface/` contains a portable adaptation of the deployed interface:

```text
RPMDrate q(3,10,B), bohr
  -> sub_250104.py (Python 2 or 3, persistent JSON workers)
     -> PES_DME_20250118.py: B configurations of the first 9 atoms
     -> PES_diff_20250224.py: B configurations of all 10 atoms
  -> sum V(B) and dVdq(3,10,B), Hartree and Hartree/bohr
```

Set the three paths in a local copy of `paths.example.sh` and source it. The model
Python must support Python 3.6+ and the TensorFlow environment. In an RPMDrate input,
import `initialize_potential` and `get_potential` from `sub_250104`, with the example
directory on `PYTHONPATH`. To test the bridge outside RPMDrate:

```bash
source /path/to/your/paths.sh
python examples/rpmd_interface/evaluate_xyz.py your_odme_frames.xyz
```

The fixed atom order is `O C H H H C H H H O`. Both source checkpoints use kcal/mol
and are external dependencies. Their hard-coded architecture options remain in the
two worker files. The DME contribution adds a zero gradient for the incoming oxygen.
The two workers execute sequentially: one complete PES call uses two batch session
executions, not one. Call `close_potential()` after use when driving the bridge yourself.
The bridge supports one outstanding request at a time and is not thread-safe.

**Legacy PES behavior retained:** the worker examples include the source deployment's
bond/angle penalty energies, but their matching gradients are not fully implemented.
They are provenance examples, not a newly validated general-purpose dynamics PES.
When those penalties activate, the returned gradient is not the derivative of the
entire returned energy. Review or replace this behavior before using these particular
workers for new production dynamics. The generic batch calculator does not add penalties.

### Changes relative to the deployed source

- Base: `MMunibas/PhysNet` commit `e243e2c383b4ac0a9d7062e654d9b4feb76ca859`.
- The deployed batch calculator was inspected on 2026-11-25. Its segment packing,
  pair-index offsets, energy/force fetch and full-pair cache are retained.
- Model files use TF1 compatibility imports; standalone and package calculator imports
  are supported. GPU memory behavior is configured externally, rather than requiring
  `tf.config` at import time (`TF_FORCE_GPU_ALLOW_GROWTH=true` on supported TF versions).
- The upstream single-frame geometry cache is retained instead of forcing every
  property access to recompute. Batch always evaluates its supplied configurations.
- Batch ensembles are explicitly rejected: the source batch route did not implement
  ensemble checkpoint iteration. Empty configurations and PBC without `lr_cut` fail clearly.
- All configurations share the calculator's total charge. The scalar charge broadcasts
  per segment. Different per-configuration total charges are not exposed by this API.
- The deployment's frozen energy/charge scale/shift initialization changes are **not**
  included in this batch extension. They concern training/model setup; checkpoint
  restoration supplies these variables during inference. Real-checkpoint comparison
  is still needed for any migrated model.
- RPMD worker paths are configurable. The bridge idle timeout is corrected to 1800
  seconds, stderr is inherited to avoid an unread pipe, and explicit close is provided.
- No checkpoints, private datasets, server account paths or trajectory logs are included.

## Original training requirements

To run this software, you need:

- python3 (tested with version 3.6.3)
- TensorFlow (tested with version 1.10.1)



## Original training usage

Edit the config.txt file to specify hyperparameters, dataset location, training/validation set size etc.
(see "train.py" for a list of all options)

Then, simply run

```
python3 train.py 
```

in a terminal to start training. 

The included "config.txt" assumes that the dataset "sn2_reactions.npz" is present. It can be downloaded from: https://zenodo.org/record/2605341. In order to use a different dataset, it needs to be formatted in the same way as this example ("sn2_reactions.npz"). Please refer to the README file of the dataset (available from https://zenodo.org/record/2605341) for details.


## How to cite

If you find this software useful, please cite:

```
Unke, O. T. and Meuwly, M. "PhysNet: A Neural Network for Predicting Energies, Forces, Dipole Moments and Partial Charges" arxiv:1902.08408 (2019).
```
