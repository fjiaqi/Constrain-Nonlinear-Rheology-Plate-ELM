# Constrain-Nonlinear-Rheology-Plate-ELM

Code for our paper:

Constraints on nonlinear mantle rheology from multi-scale geodetic observations of plate motion and post-seismic enhanced landward motion, by Jiaqi Fang, Michael Gurnis and Rishav Mallick.

## Dependencies

See `environment.yml`. Key requirements include:

* Python 3.7
* Underworld 2.10.0b

The code is expected to work well with other recent versions.

## How to run

Run the 2D model:
```
mpirun -n 64 python 2d_test.py
```

Convert 2D results to 3D initial guess:
```
python convert_initial_guess.py
```

Run the 3D model:
```
mpirun -n 1024 python param_impose_model_all {A} {V} {Gw} {Ge}
```

`{A}`, `{V}`, `{Gw}`, and `{Ge}` are parameter values required by the script.

The 3D model (1024 × 128 × 256) runs on 1024 MPI ranks on the Anvil supercomputer at Purdue University.

## Post-processing

Extract key results:
```
python3 extract_key_res.py {directory}
```

`{directory}` is the directory containing model outputs.

Key figures in the paper can be plotted using `make_res_figs.py` based on extracted results. Additional figures can be plotted with `make_model_figs.py` and `make_other_figs.py` based on model outputs.
