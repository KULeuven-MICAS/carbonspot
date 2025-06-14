# CarbonSpot

This repository presents the CarbonSpot, a analytical design space exploration (DSE) framework evaluating the performance and the carbon footprint for AI accelerators.
CarbonSpot bridges the gap between the performance-oriented DSE framework and the carbon-oriented DSE framework, enabling the architecture comparison between the most performant hardware choice and the most carbon-friendly hardware choice within one framework.

The part of the performance simulator is inherited from our previous framework [ZigZag-IMC](https://github.com/KULeuven-MICAS/zigzag-imc) and [ZigZag](https://github.com/KULeuven-MICAS/zigzag), with analytical accelerators performance models that have been validated against chip results.

If you find this repository useful to your work, please consider cite our paper in your study: [TBD]

<!-- ```bibtex
@inproceedings{carbonspot2025,
    title={A Unified Analytical Model for Performance-Carbon Co-Optimization of Edge AI Accelerators},
    author={Jiacong Sun, Xiaoling Yi, Arne Symons, Georages Gielen, Lieven Eeckhout, Marian Verhelst},
    booktitle={Proceedings of the 2024 Conference on Example},
    year={2025}
}
``` -->

## Motivation

The size of current AI models is increasing drastically in recent years together with model accuracy and capability. We plotted this trend in the figure below (data is collected from [here](https://paperswithcode.com/sota/image-classification-on-imagenet)). At the same time, this demands powerful hardware and significant energy consumption, bringing our community attention to evaluate the environmental impacts -- or more specific, the equivalent carbon footprint.

<p align="center">
    <img src="https://github.com/KULeuven-MICAS/carbonspot/blob/master/model_size_trends.png" width="80%" alt="The trend of increasing model size and accuracy">
</p>

## Framework Capability

CarbonSpot is capabile of:

- Figuring out the optimal mapping for a user-defined accelerator architectures, including digital TPU-like architecture, analog In-Memory-Computing architecture and digital In-Memory-Computing architecture.
- Reporting the energy, latency and carbon footprint for a given architecture and workload.
- Comparing the optimal architectures in terms of performance and in terms of carbon footprint, with respective optimal mapping.
- Evaluating the carbon footprint under Continuous-Active (CA) scenario and Periodically-Active (PA) scenario.
- Reporting the cost breakdown for energy, latency and carbon footprint.

MLPerf-Tiny and MLPerf-Mobile wokloads are placed under folder [./zigzag/inputs/examples/workload/](https://github.com/KULeuven-MICAS/carbonspot/tree/master/zigzag/inputs/examples/workload). The timing requirement for each workload are summarized below.

If you find this table useful to your work, please consider cite our paper in your study!

| Workload Suite                                                         | Network Name | Usecase                     | Targetd Dataset           | Workload Size (MB) | Frame/Second Requirement | Paper Reference |
|------------------------------------------------------------------------|--------------|-----------------------------|---------------------------|--------------------|--------------------------|-----------------|
| [MLPerf-Tiny](https://github.com/mlcommons/tiny/tree/master/benchmark) | DS-CNN       | Keyword Spotting            | Speech Commands           | 0.06               | 10                       | [[1]](https://arxiv.org/abs/1804.03209) |
| [MLPerf-Tiny](https://github.com/mlcommons/tiny/tree/master/benchmark) | MobileNet-V1 | Visual Wake Words           | Visual Wake Words Dataset | 0.9                | 0.75                     | [[2]](https://arxiv.org/abs/1906.05721) |
| [MLPerf-Tiny](https://github.com/mlcommons/tiny/tree/master/benchmark) | ResNet8      | Binary Image Classification | Cifar10                   | 0.3                | 25                       | [[3]](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf), [[4]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Cross-View_Tracking_for_Multi-Human_3D_Pose_Estimation_at_Over_100_CVPR_2020_paper.pdf) |
| [MLPerf-Tiny](https://github.com/mlcommons/tiny/tree/master/benchmark) | AutoEncoder  | Anamaly Detection           | ToyADMOS                  | 1.0                | 1                        | [[5]](https://arxiv.org/abs/1909.09347) |
| [MLPerf-Mobile](https://github.com/mlcommons/mobile_models/tree/main/v0_7/tflite) | MobileNetV3           | Image Classification            | ImageNet                     | 15.6              | 25*                       | [[6]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf) |
| [MLPerf-Mobile](https://github.com/mlcommons/mobile_models/tree/main/v0_7/tflite) | SSD MobileNetV2       | Object Classification           | COCO                         | 64.4              | 25*                       | [[7]](https://vidishmehta204.medium.com/object-detection-using-ssd-mobilenet-v2-7ff3543d738d) |
| [MLPerf-Mobile](https://github.com/mlcommons/mobile_models/tree/main/v0_7/tflite) | DeepLab MobileNetV2   | Semantic Segmentation           | ImageNet ADE20K Training Set | 8.7               | 25*                       | [[8]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) |
| [MLPerf-Mobile](https://github.com/mlcommons/mobile_models/tree/main/v0_7/tflite) | MobileBert            | Language Understanding          | NA                           | 96                | 25*                       | [[9]](https://arxiv.org/abs/2004.02984) |

*: 25 FPS is borrowed from the setting for ResNet8 as this information is missing in the paper reference.

## Getting Started

To get started, you can install all packages directly through pip using the pip-requirements.txt with the command:

`$ pip install -r requirements.txt`

The main script is `expr.py`, which can:

### Evaluate the carbon footprint of prior works in literature

The function `experiment_1_literature_trend()` can output the equivalent carbon footprint of chips reported in prior works. Following graphs can be generated.

<p align="center">
    <img src="https://github.com/KULeuven-MICAS/carbonspot/blob/master/literature_carbon.png" width="100%" alt="Carbon cost (y axis) versus energy efficiency (x axis) of AI accelerators from the literature in 16-28 nm CMOS technology when applied on the (a) MLPerf-Tiny and (b) MLPerf-Mobile benchmarks. The pie charts show the carbon breakdown into operational (green) and embodied (red) carbon costs for each design. TOP/s/W has been normalized to INT8. The most performant and most carbon-efficient designs are circled out.">
</p>
<p align="center"><b>Figure 1:</b> Carbon cost (y axis) versus energy efficiency (x axis) of AI accelerators from the literature in 16-28 nm CMOS technology when applied on the (a) MLPerf-Tiny and (b) MLPerf-Mobile benchmarks. The pie charts show the carbon breakdown into operational (green) and embodied (red) carbon costs for each design. TOP/s/W has been normalized to INT8. The most performant and most carbon-efficient designs are circled out.</p>

Follwing papers are used to generate the data points in the figure. Due to the page limitation, please forgive us not able to include the citations in the paper. We here sincerely want to express our gratitude to these amazing works.

[1] [Chih, Yu-Der, et al. "16.4 An 89TOPS/W and 16.3 TOPS/mm 2 all-digital SRAM-based full-precision compute-in memory macro in 22nm for machine-learning edge applications." 2021 IEEE International Solid-State Circuits Conference (ISSCC). Vol. 64. IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9365766)

[2] [Tu, Fengbin, et al. "A 28nm 15.59 µJ/token full-digital bitline-transpose CIM-based sparse transformer accelerator with pipeline/parallel reconfigurable modes." 2022 IEEE International Solid-State Circuits Conference (ISSCC). Vol. 65. IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9731645)

[3] [Lee, Chia-Fu, et al. "A 12nm 121-TOPS/W 41.6-TOPS/mm2 all digital full precision SRAM-based compute-in-memory with configurable bit-width for AI edge applications." 2022 IEEE Symposium on VLSI Technology and Circuits (VLSI Technology and Circuits). IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9830438)

[4] [Wang, Dewei, et al. "DIMC: 2219TOPS/W 2569F2/b digital in-memory computing macro in 28nm based on approximate arithmetic hardware." 2022 IEEE international solid-state circuits conference (ISSCC). Vol. 65. IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9731659)

[5] [Jiang, Weijie, et al. "A 16nm 128kB high-density fully digital In Memory Compute macro with reverse SRAM pre-charge achieving 0.36 TOPs/mm 2, 256kB/mm 2 and 23. 8TOPs/W." ESSCIRC 2023-IEEE 49th European Solid State Circuits Conference (ESSCIRC). IEEE, 2023.](https://ieeexplore.ieee.org/abstract/document/10268774)

[6] [Jia, Hongyang, et al. "15.1 a programmable neural-network inference accelerator based on scalable in-memory computing." 2021 IEEE International Solid-State Circuits Conference (ISSCC). Vol. 64. IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9365788)

[7] [Liu, Shiwei, et al. "16.2 A 28nm 53.8 TOPS/W 8b sparse transformer accelerator with in-memory butterfly zero skipper for unstructured-pruned NN and CIM-based local-attention-reusable engine." 2023 IEEE International Solid-State Circuits Conference (ISSCC). IEEE, 2023.](https://ieeexplore.ieee.org/abstract/document/10067360)

[8] [Zhao, Yuanzhe, et al. "A double-mode sparse compute-in-memory macro with reconfigurable single and dual layer computation." 2023 IEEE Custom Integrated Circuits Conference (CICC). IEEE, 2023.](https://ieeexplore.ieee.org/abstract/document/10121308)

[9] [Lin, Chuan-Tung, et al. "iMCU: A 102-μJ, 61-ms Digital In-Memory Computing-based Microcontroller Unit for Edge TinyML." 2023 IEEE Custom Integrated Circuits Conference (CICC). IEEE, 2023.](https://ieeexplore.ieee.org/abstract/document/10121221)

[10] [Lee, Jinseok, et al. "Fully row/column-parallel in-memory computing SRAM macro employing capacitor-based mixed-signal computation with 5-b inputs." 2021 Symposium on VLSI Circuits. IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9492444)

[11] [Yin, Shihui, et al. "PIMCA: A 3.4-Mb programmable in-memory computing accelerator in 28nm for on-chip DNN inference." 2021 Symposium on VLSI Technology. IEEE, 2021.](https://ieeexplore.ieee.org/abstract/document/9508673)

[12] [Zhu, Haozhe, et al. "COMB-MCM: Computing-on-memory-boundary NN processor with bipolar bitwise sparsity optimization for scalable multi-chiplet-module edge machine learning." 2022 IEEE International Solid-State Circuits Conference (ISSCC). Vol. 65. IEEE, 2022.](https://ieeexplore.ieee.org/abstract/document/9731657)

[13] [Song, Jiahao, et al. "A 28 nm 16 kb bit-scalable charge-domain transpose 6T SRAM in-memory computing macro." IEEE Transactions on Circuits and Systems I: Regular Papers 70.5 (2023): 1835-1845.](https://ieeexplore.ieee.org/abstract/document/10044587)

### Simulate and estimate the performance and carbon footprint of user-defined accelerators

The function `zigzag_similation_and_result_storage()` simulates and evalutes the performance and carbon footprint for given architectures. An example demonstration is shown below.

<p align="center">
    <img src="https://github.com/KULeuven-MICAS/carbonspot/blob/master/carbon_example.png" width="80%" alt="An example demonstration of CarbonSpot output">
</p>
<p align="center"><b>Figure 2:</b> An example demonstration of CarbonSpot output. The figures show the carbon footprint under PA scenario (y axis) and CA scenario (x axis) across different architecture solutions. Different colors mean different SRAM size. (enable <b>active_plot=True</b> to see which point corresponds to what architecture solution.) </p>


## Performance Model Description

Note in the CarbonSpot paper, we only show the simulation results on digital TPU-like architecture and digital In-Memory-Computing architectures. In fact, the framework supports all propagation-based digital architectuers with any random data stationarity dataflow, and analog In-Memory-Computing architectures.

Our SRAM-based In-Memory-Computing performance model is borrowed from [ZigZag-IMC](https://github.com/KULeuven-MICAS/zigzag-imc), which supports both analog (AIMC) and digital (DIMC) In-Memory-Computing architectures.
A summary of the hardware settings for these chips is provided in the following table.

| source                                                      | label | B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub> | macro size     | #cell_group | nb_of_macros |
|-------------------------------------------------------------|-------|-----------------------------------------------|----------------|-------------|--------------|
| [[1]](https://ieeexplore.ieee.org/abstract/document/9431575)  | AIMC1 | 7 / 2 / 7                                     | 1024&times;512 | 1           | 1            |
| [[2]](https://ieeexplore.ieee.org/abstract/document/9896828)  | AIMC2 | 8 / 8 / 2                                     | 16&times;12    | 32          | 1            |
| [[3]](https://ieeexplore.ieee.org/abstract/document/10067289) | AIMC3 | 8 / 8 / 1                                     | 64&times;256   | 1           | 8            |
| [[4]](https://ieeexplore.ieee.org/abstract/document/9731762)  | DIMC1 | 8 / 8 / 2                                     | 32&times;6     | 1           | 64           |
| [[5]](https://ieeexplore.ieee.org/abstract/document/9731545)  | DIMC2 | 8 / 8 / 1                                     | 32&times;1     | 16          | 2            |
| [[6]](https://ieeexplore.ieee.org/abstract/document/10067260) | DIMC3 | 8 / 8 / 2                                     | 128&times;8    | 8           | 8            |
| [[7]](https://ieeexplore.ieee.org/abstract/document/10067779) | DIMC4 | 8 / 8 / 1                                     | 128&times;8    | 2           | 4            |

B<sub>i</sub>/B<sub>o</sub>/B<sub>cycle</sub>: input precision/weight precision/number of bits processed per cycle per input.
#cell_group: the number of cells sharing one entry to computation logic.

The validation details can be found at [here](https://github.com/KULeuven-MICAS/zigzag-imc).

Our digital performance model is based on [ZigZag](https://github.com/KULeuven-MICAS/zigzag). The validation details can be found at [here](https://kuleuven-micas.github.io/zigzag/hardware.html).


## Carbon Model Description

The carbon model is developed based upon [ACT](https://dl.acm.org/doi/abs/10.1145/3470496.3527408). To estimate the carbon footprint of prior works, we adapt its equations and derive the following equations (used in `expr.py`):

For Continuous-Active (CA) scenario:

$Carbon/operation = \frac{k_1}{TOP/s/W} + \frac{k_2}{TOP/s/mm^2} + package\ cost$

For Periodic-Active (PA) scenario:

$Carbon/operation = \frac{k_1}{TOP/s/W} + \frac{k_2 \cdot T_{c} \cdot TOP/s}{TOP/s/mm^2 \cdot parallelism} + package\ cost$

where, $k_1$ is the operational carbon intensity ($\frac{301}{3.6E+18}\ g, CO_2/pJ$ in globe average).
$k_2$ is the embodied carbon intensity ($8.709 \cdot \frac{1}{Yield \cdot lifetime(year)}\ g, CO_2/mm^2/ps$).
$T_c$ is the reponse time constraint under PA scenario.
