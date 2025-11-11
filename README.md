# Gearbox Fault Diagnosis with MSSR–BNN

This repository contains a reference implementation for the paper
"**An Uncertainty‑Aware Combined Multistable Stochastic Resonance Bayesian Neural Network for Gearbox Fault Diagnosis in Noisy Environments**".  

The goal of this project is to reproduce the key experiments described in the
paper and provide a clean, modular code base that other researchers can use
to explore vibration‑based fault diagnosis.  The implemented pipeline uses
synthetic data that mimics vibration signals measured from gearboxes under
different fault conditions and noise levels.  A set of classical
machine–learning models (SVM, Random Forest), deep learning models (CNN,
LSTM, Transformer, GCN and ResNet) and the proposed MSSR‑BNN model are
implemented and evaluated using accuracy, F1 score, expected calibration
error (ECE), negative log‑likelihood (NLL) and Brier score.

## Repository Structure

```
gearbox_fault_diagnosis/
├── data/
│   ├── generate_data.py    # Script to generate synthetic vibration signals
│   └── dataset.npz         # Example synthetic dataset (generated on first run)
├── models/
│   ├── __init__.py
│   ├── cnn.py             # Convolutional Neural Network
│   ├── lstm.py            # Long Short‑Term Memory network
│   ├── transformer.py     # Simple Transformer encoder
│   ├── resnet.py          # 1D ResNet implementation
│   ├── gcn.py             # Simple Graph Convolutional Network
│   └── mssr_bnn.py        # Multistable Stochastic Resonance + Bayesian NN
├── utils/
│   ├── metrics.py         # Evaluation metrics (accuracy, F1, ECE, NLL, Brier)
│   └── train_utils.py     # Training and evaluation helpers
├── train.py               # Entry script for training/evaluation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Getting Started

### Prerequisites

This project requires Python 3.8+ and the following packages:

- numpy
- scikit‑learn
- torch (PyTorch)
- tqdm
- matplotlib (optional, for plotting)

You can install the dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Alternatively, if you prefer using a conda environment, you can create
a new environment and install the dependencies manually:

```bash
conda create -n gearbox python=3.8
conda activate gearbox
pip install -r requirements.txt
```

### Data Generation

The repository does not include proprietary datasets.  Instead it uses a
synthetic dataset that simulates vibration signals from gearboxes under
various fault conditions.  To generate the synthetic dataset, run:

```bash
python data/generate_data.py
```

This command will create a file `data/dataset.npz` containing training and
test splits (`X_train`, `y_train`, `X_test`, `y_test`).  By default the
script generates 2000 samples with four classes (healthy, broken tooth,
wear and defect).  Each sample is a one‑second signal sampled at 1 kHz.
Noise with randomly selected signal‑to‑noise ratios between 0 and 20 dB
is added to emulate different operating conditions.  You can adjust the
number of samples and classes in `data/generate_data.py`.

### Training a Single Model

Use the `train.py` script to train and evaluate a model.  The script
accepts several command‑line arguments:

```bash
python train.py --model cnn --epochs 10 --batch-size 32
```

The `--model` argument selects which classifier to use.  Supported
values are:

* `svm` – Support Vector Machine (scikit‑learn)
* `rf` – Random Forest (scikit‑learn)
* `cnn` – Convolutional Neural Network
* `lstm` – Long Short‑Term Memory network
* `transformer` – Transformer encoder
* `gcn` – Graph Convolutional Network
* `resnet` – 1D ResNet
* `mssr_bnn` – Proposed MSSR‑BNN model
* `all` – Train and evaluate all of the above models in sequence

For PyTorch models (cnn, lstm, transformer, gcn, resnet and mssr_bnn)
the script trains for the specified number of epochs and reports
metrics on the test set.  Scikit‑learn models (svm and rf) ignore the
`--epochs` and `--batch-size` arguments because they do not use
mini‑batch training.

### Running All Experiments

You can benchmark all models at once by setting `--model all`:

```bash
python train.py --model all --epochs 10 --batch-size 64
```

This command generates the dataset if necessary, trains each model in
turn and prints a summary of accuracy, macro F1, ECE, NLL and Brier
score for every model.  Training times are also reported.

### Expected Output

After training, the script will output accuracy, F1 score, expected
calibration error (ECE), negative log‑likelihood (NLL) and Brier score
for each model.  You can use these results to compare the performance of
the proposed MSSR‑BNN against baseline methods.  Note that due to the
synthetic nature of the dataset and small network configurations used in
this reference implementation, the absolute values of these metrics may
not match those reported in the paper, but the relative trends (e.g.,
MSSR‑BNN outperforming other models) should be similar.

### Troubleshooting

**PyTorch not installed or GPU unavailable** – Ensure that PyTorch is
installed via `pip install torch` or install the CPU version if a GPU
is unavailable.  The code automatically detects whether a GPU is
available and uses it if possible.

**Dataset generation fails** – If you encounter errors when running
`generate_data.py`, ensure that the `data` directory exists and you
have write permissions.  You can create the directory manually via
`mkdir -p data`.

**Training is slow** – The deep models in this repository are
intentionally small to run quickly on CPUs.  If training takes too
long, reduce the number of epochs (e.g., `--epochs 5`) or decrease the
number of samples generated in `generate_data.py`.

## Repository Contents

### data/generate_data.py

Generates a synthetic dataset of vibration signals.  Each sample is
simulated as a sum of sine waves plus Gaussian noise.  The script
splits the data into training and testing sets and saves it as a
NumPy archive in `data/dataset.npz`.

### models/

Contains the model definitions.  Each model is implemented as a
PyTorch `nn.Module`:

* `cnn.py` – A simple convolutional network with two convolutional layers,
  batch normalisation and dropout.
* `lstm.py` – A sequence model using a single LSTM layer followed by
  fully connected layers.
* `transformer.py` – A minimal Transformer encoder with positional
  encoding.
* `gcn.py` – A basic graph convolution network that operates on
  down‑sampled signals to construct a k‑nearest neighbour graph.
* `resnet.py` – A 1D version of a residual network with skip
  connections.
* `mssr_bnn.py` – Implements the proposed MSSR‑BNN.  It uses three
  independent CNN experts with Monte Carlo dropout to approximate
  Bayesian inference and a gating network to combine their outputs.

### utils/

Helper functions for computing metrics and training models:

* `metrics.py` – Functions to compute accuracy, F1 score, ECE, NLL and
  Brier score.
* `train_utils.py` – Functions to load data, create data loaders, and
  perform training loops for PyTorch models.  Also contains helper
  functions to train scikit‑learn models.

### train.py

This is the main script used to run experiments.  It parses
command‑line arguments, loads the dataset, constructs the specified
model, trains it and evaluates its performance.  Results are printed
to the console for easy comparison across models.

## Citation

If you use this code or dataset, please cite the original paper:

> Peiliang Qin, Rongjing Hong.  "An Uncertainty‑Aware Combined
> Multistable Stochastic Resonance Bayesian Neural Network for Gearbox
> Fault Diagnosis in Noisy Environments." IEEE Access. 2024.

This repository and the synthetic dataset are provided for research and
educational purposes only.  For commercial applications or use with
proprietary data, please ensure compliance with all relevant
regulations and guidelines.
