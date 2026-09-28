# Installation

Tested on Ubuntu 24.04 with an NVIDIA A100, CUDA 12.1, PyTorch 2.5.1, Python 3.10.
Any recent Linux + CUDA setup with PyTorch >= 2.0 should work; CPU works for
evaluation but pretraining is slow.

## Environment

```bash
conda create -n oswb python=3.10 -y
conda activate oswb
pip install -r requirements.txt
```

## SMPL model (required for FK and synthetic data)

The forward-kinematics layer and the AMASS synthesizer need the official SMPL
male model, which is license-gated:

1. Register and download **SMPL v1.0.0** from https://smpl.is.tue.mpg.de/
   (the "SMPL for Python" package).
2. Place the male model at:

```
data/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

## AMASS (required to generate synthetic pretraining data)

1. Register and download **AMASS** from https://amass.is.tue.mpg.de/ — the
   sub-datasets `CMU`, `BioMotionLab_NTroje`, and `MPI_HDM05` (SMPL+H G format).
2. Arrange them so the synthesizer finds `data/amass/<Dataset>/*/*_poses.npz`:

```
data/amass/CMU/01/01_01_poses.npz
data/amass/BioMotionLab_NTroje/...
data/amass/MPI_HDM05/...
```

3. Generate the virtual head/foot IMU tensors (one-time, ~1.5 GB output):

```bash
python synth_amass.py
```

## Benchmark data (35 takes)

The captured benchmark (aligned earbud + insole IMU streams with pseudo-GT
labels, `data/processed/run*_seq*.npz`, ~90 MB) will be released separately;
see the README. Once downloaded, unpack so that:

```
data/processed/run1_seq1.npz ... run5_seq7.npz
```

Without it you can still run synthetic pretraining; all real-data evaluation
and fine-tuning steps require it.
