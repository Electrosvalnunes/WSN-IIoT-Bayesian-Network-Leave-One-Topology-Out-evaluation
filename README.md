# WSN-IIoT-Bayesian-Network-Leave-One-Topology-Out-evaluation
Topology-aware Bayesian cyberattack detection in WSN-based IIoT using OMNeT++/INET simulations, intra-topology validation, simple LOTO, and calibrated LOTO protocols.



This repository provides the code, simulation files, and reproducibility materials used in the study on topology-aware cyberattack detection in WSN-based Industrial Internet of Things (IIoT) networks.

The work evaluates a discrete Bayesian detector for identifying five operating conditions: Normal, Flooding, Blackhole, Wormhole, and Manipulated Backoff. The experiments are based on OMNeT++/INET simulations using IEEE 802.15.4, 6LoWPAN, RPL, and UDP communication protocols.

The main objective is to analyze how the Bayesian detector generalizes when tested on network topologies that were not observed during training. For this purpose, the repository includes scripts for intra-topology validation, simple Leave-One-Topology-Out (LOTO), and Normal-baseline calibrated LOTO evaluation.

## Repository structure

```text
 WSN-IIoT-Bayesian-Network-Leave-One-Topology-Out-evaluation/
│
├── Modelagem OMNeT/
│   ├── 01_omentpp_36.ini
│   ├── 02_omentpp_49.ini
│   ├── 03_omentpp_64.ini
│   ├── 04_omentpp_100.ini
│   ├── 05_networks_36.ned
│   └── 06_networks_49.ned
│   ├── 07_networks_64.ned
│   └── 08_networks_100.ned
│  
├── Dataset/
│   └── processed/
│             └── Dataset_Expermental_20Mil.csv
├── models/
│   └── RB_Q5.xdsl
├── scripts/
│   ├── 01_protocolo_1_intra_topology.py
│   ├── 02_protocolo_2_simple_loto.py
│   └── 03_protocolo_3_calibrated_loto.py
│     
│
├── results/
│   ├── tables/
│   └── figures/
│
│
├── requirements.txt
├── LICENSE
└── README.md
```

## Simulation environment

The simulation scenarios were configured using:

* OMNeT++ 5.7.1;
* INET 4.2.1;
* IEEE 802.15.4 MAC;
* 6LoWPAN/IPv6 stack;
* RPL routing protocol;
* UDP application traffic.

The simulated topologies include grid networks with 36, 49, 64, and 100 nodes. The evaluated classes are:

* Normal;
* Flooding;
* Blackhole;
* Wormhole;
* Manipulated Backoff.

## Dataset

The processed dataset used to reproduce the experiments is available in the `data/processed/` directory.

The dataset is also available on Zenodo:

https://zenodo.org/records/20602871


## Bayesian Network model

The `models/RB_Q5.xdsl` file contains the discrete Bayesian Network model used in the study. The model was created in GeNIe/SMILE format and represents the probabilistic relationships between topology, attack type, and the discretized network metrics: PDR, delay, throughput, and energy. Each metric is represented using five qualitative states: Very Low, Low, Medium, High, and Very High.

## Evaluation protocols

The `scripts/` directory contains the Python implementations of the three evaluation protocols used in the study:

* `01_evaluate_intra_topology.py`: implements the intra-topology validation protocol, where training and testing are performed within the same network topology.
* `02_evaluate_loto_simple.py`: implements the simple Leave-One-Topology-Out protocol, where the model is trained on three topologies and tested on the unseen topology.
* `03_evaluate_loto_calibrated.py`: implements the Normal-baseline calibrated LOTO protocol, where Normal samples from the unseen topology are used to estimate local baseline statistics before cyberattack detection.

Each script computes accuracy, macro-precision, macro-recall, macro-F1, ROC-AUC, confusion matrices, and ROC curves.



## Installation

The Python scripts require Python 3.x and the packages listed in `requirements.txt`.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```text
numpy
pandas
scikit-learn
matplotlib
```

The Bayesian inference protocols implemented in this repository do not require `pgmpy`, since the conditional probability tables, Laplace smoothing, MAP inference, evaluation metrics, confusion matrices, and ROC curves are implemented directly in Python.

## Bayesian Network model file

The repository also includes the Bayesian Network model file:

```text
models/RB_Q5.xdsl
```

This file can be opened with GeNIe/SMILE and provides the graphical/probabilistic representation of the discrete Bayesian Network used in the study. It is provided for model inspection, documentation, and reproducibility purposes, but it is not required to execute the Python evaluation scripts.


## Citation

If you use this repository, the Bayesian model, or the associated dataset, please cite the related paper and the dataset record available on Zenodo:

https://zenodo.org/records/20602871

A formal citation will be added after publication of the paper.

## License

The source code in this repository is distributed under the MIT License. See the `LICENSE` file for more details.

The dataset is distributed through Zenodo and follows the license specified in the corresponding Zenodo record.
