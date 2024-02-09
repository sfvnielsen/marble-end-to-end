# End-to-end learning of optic fiber communication systems
This repository contains a collection of Python implementations for learning optimal transmitter and receiver parameters in an optical communication system. The term end-to-end refers to the fact that parameters from both sides of the channel are jointly optimized.

Currently, the repo uses automatic differentiation via. `pytorch` for ease of experimentation and prototyping.

## Getting started
The list of required packages is collected in the `requirements.txt` file. The Python version used to create the environment is 3.10.12.
Given that a Python virtual environment has been created (or an equivalent conda environment) the environment can be installed by the following command

```
pip install -r requirements.txt
```

Afterwards, install `torch` and `torchaudio` to fit your system specifications (CPU, GPU, Cuda version).
Please refer to the [pytorch](https://pytorch.org/get-started/locally/) installation page for details on this. 

Once installed you should be able to run the main optimization script

```
python main_e2e.py
```

## Acknowledgements
The work carried out in this repository is part of a research project [MAchine leaRning enaBLEd fiber optic communication](https://veluxfoundations.dk/en/villum-synergy-2021) (MARBLE) funded by the Villum foundation.

## References

[1] B. Karanov et al., “End-to-End Deep Learning of Optical Fiber Communications,” Journal of Lightwave Technology, vol. 36, no. 20, pp. 4843–4855, Oct. 2018.

[2] O. Jovanovic, M. P. Yankov, F. Da Ros, and D. Zibar, “End-to-End Learning of a Constellation Shape Robust to Channel Condition Uncertainties,” Journal of Lightwave Technology, vol. 40, no. 10, pp. 3316–3324, May 2022, doi: 10.1109/JLT.2022.3169993.

