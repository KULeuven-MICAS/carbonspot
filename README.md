# CarbonSpot

This repository presents the CarbonSpot, a analytical design space exploration (DSE) framework evaluating the performance and the carbon footprint for AI accelerators.
CarbonSpot bridges the gap between the performance-oriented DSE framework and the carbon-oriented DSE framework, enabling the architecture comparison between the most performant hardware choice and the most carbon-friendly hardware choice within one framework.

The framework is built upon our previous framework [ZigZag-IMC](https://github.com/KULeuven-MICAS/zigzag-imc) and [ZigZag](https://github.com/KULeuven-MICAS/zigzag), which inherit analytical accelerators models that have been validated against chip results.

## Framework Capability

xx

## Prerequisite

To get started, you can install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r requirements.txt`

## Getting Started
The main script is `expr.py`, which can:

### Evaluate the carbon footprint of prior works in literature

xxx

### Simulate and estimate the performance and carbon footprint of user-defined accelerators

xxx


## In-Memory Computing AI Accelerator Cost Model Description
Our SRAM-based In-Memory Computing model is a versatile, parameterized model designed to cater to both Analog IMC and Digital IMC.
Since hardware costs are technology-node dependent, we have performed special calibration for the 28nm technology node. The model has been validated against 7 chips from the literature. 
A summary of the hardware settings for these chips is provided in the following table.

| source                                                          | label | B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub> | macro size     | #cell_group | nb_of_macros |
|-----------------------------------------------------------------|-------|-----------------------------------------------|----------------|-------------|--------------|
| [paper](https://ieeexplore.ieee.org/abstract/document/9431575)  | AIMC1 | 7 / 2 / 7                                     | 1024&times;512 | 1           | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9896828)  | AIMC2 | 8 / 8 / 2                                     | 16&times;12    | 32          | 1            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067289) | AIMC3 | 8 / 8 / 1                                     | 64&times;256   | 1           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731762)  | DIMC1 | 8 / 8 / 2                                     | 32&times;6     | 1           | 64           |
| [paper](https://ieeexplore.ieee.org/abstract/document/9731545)  | DIMC2 | 8 / 8 / 1                                     | 32&times;1     | 16          | 2            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067260) | DIMC3 | 8 / 8 / 2                                     | 128&times;8    | 8           | 8            |
| [paper](https://ieeexplore.ieee.org/abstract/document/10067779) | DIMC4 | 8 / 8 / 1                                     | 128&times;8    | 2           | 4            |

B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub>: input precision/weight precision/number of bits processed per cycle per input.
#cell_group: the number of cells sharing one entry to computation logic.

The validation results are displayed in the figure below (assuming 50% input toggle rate and 50% weight sparsity are assumed). 
The gray bar represents the reported performance value, while the colored bar represents the model estimation.
The percent above the bars is the ratio between model estimation and the chip measurement results.

<p align="center">
<img src="https://github.com/KULeuven-MICAS/carbonspot/blob/master/imc_model_validation/model_validation.png" width="100%" alt="imc model validation plot">
</p>

## Digital AI Accelerator Cost Model Description

xx


