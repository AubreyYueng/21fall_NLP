# Fall 2021 CSE 538 - Natural Language Processing Assignment 3 - Transition Parsing with Neural Networks

This file is best viewed in Markdown reader (eg. https://jbt.github.io/markdown-editor/)

# Credits and Disclaimer

**Credits**: Most of this code has been directly taken from the authors of:

> From A Fast and Accurate Dependency Parser using Neural Networks (2014, Danqi and Manning)

This assignment has been adapted, implemented and revamped as required by many NLP TAs to varying degrees. In chronological order of TAship they include Heeyoung Kwon, Jun Kang, Mohaddeseh Bastan, Harsh Trivedi, Matthew Matero, Yash Kumar Lal, Sounak Mondal. Thanks to all of them!

**Disclaimer/License**: This code is only for school assignment purpose, and **any version of this should NOT be shared publicly on github or otherwise even after semester ends**. Public availability of answers devalues usability of the assignment and work of several TAs who have contributed to this. We hope you'll respect this restriction.

# Overview

In this assignment, you will implement the Neural Network based Transition Parsing system described in [Chen and Manning, 2014](https://aclanthology.org/D14-1082/). This assignment con- sists of following components:

## Parsing Algorithm

In `parsing system.py`, there is an overall wrapper that begins with the initial state, and asks the classifier to provide a distribution over the available actions, selects the next available legal action, and then changes the state by applying the action.

You will be implementing the ARC standard system rather than the eager version. This is a slight modification with only three types of actions (shift, left, right) instead of the four. [Nivre, 2004](https://aclanthology.org/W04-0308/) describes this in detail.

## Neural Network Classifier using PyTorch

You will implement the training and test functionality for the transition clas- sifier. You will generate the features given the (sentence, tree) pairs and write the loss function to use for training the network.

## Code Requirements

You will implement a neural Dependency Parsing model by writing code for the following:

From Incrementality in Deterministic Dependency Parsing [Nivre, 2004](https://aclanthology.org/W04-0308/)
- the arc-standard algorithm

From A Fast and Accurate Dependency Parser using Neural Networks [Chen and Manning, 2014](https://aclanthology.org/D14-1082/)
- feature extraction
- the neural network architecture including activation function
- loss function

# Installation

This assignment is implemented in python 3.6 and PyTorch 1.9. Follow these steps to setup your environment:

1. [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a Conda environment with Python 3.6
```
conda create -n nlp-hw3 python=3.6
```
3. Activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to use this code.
```
conda activate nlp-hw3
```
4. Install the requirements:
```
pip install -r requirements.txt
```
5. Download glove wordvectors:
```
./download_glove.sh
```

**NOTE:** We will be using this environment to check your code, so please don't work in your default or any other python environment.

# Data

You have training, development and test set for dependency parsing in conll format. The `train.conll` and `dev.conll` are labeled whereas `test.conll` is unlabeled.

For quick code development/debugging, this time we have explicitly provided a small `fixture.conll` dataset. You can use this as training and development dataset while working on the code.

# Code Overview

## File Structure

When you download the assignment from Blackboard, it will unzip into the following directory structure:

```
538_HW3
├── .gitignore
└── data/
  ├── train.conll
  ├── dev.conll
  ├── test.conll
  └── fixture.conll
└── lib/
  ├── __init__.py
  ├── configuration.py
  ├── data.py
  ├── dependency_tree.py
  ├── model.py
  ├── parsing_system.py
  ├── util.py
  └── fixture.conll
├── constants.py
├── download_glove.sh
├── evaluate.py
├── experiments.sh
├── predict.py
├── README.md
├── requirements.txt
└── train.py
```

## Dependency Parsing Files

  - `lib.model:` Defines the main model class of the neural dependency parser.

  - `lib.data.py`: Code dealing with reading, writing connl dataset, generating batches, extracting features and loading pretrained embedding file.

  - `lib.dependency_tree.py`: The dependency tree class file.

  - `lib.parsing_system.py`: This file contains the class for a transition-based parsing framework for dependency parsing.

  - `lib.configuration.py`: The configuration class file. Configuration reflects a state of the parser.

  - `lib.util.py`: This file contain function to load pretrained Dependency Parser.

  - `constants.py`: Sets project-wide constants for the project.

## Example Commands

You have three scripts in the repository `train.py`, `predict.py` and `evaluate.py` for training, predicting and evaluating a Dependency Parsing Model. You can supply `-h` flag to each of these to figure out how to use these scripts.

Here we show how to use the commands with the defaults.

#### Train a model
```
python train.py data/train.conll data/dev.conll

# stores the model by default at : serialization_dirs/default
```

#### Predict with model
```
python predict.py serialization_dirs/default \
                  data/dev.conll \
                  --predictions-file dev_predictions.conll
```

#### Evaluate model predictions

```
python evaluate.py serialization_dirs/default \
                   data/dev.conll \
                   dev_predictions.conll
```

**NOTE:** These scripts will not work until you fill-up the placeholders (TODOs) left out as part of the assignment.

# Expectations

## What to write in code:

Like assignment 2 you have `TODO(Students) Start` and `TODO(Students) End` annotations. You are expected to write your code between those comment/annotations.

1. Implement the arc-standard algorithm in `parsing_system.py`: `apply` method
2. Implement feature extraction in `data.py`: `get_configuration_features` method
3. Implement neural network architecture in `model.py` in `DependencyParser` class: `__init__` and `call` method.
4. Implement loss function for neural network in `model.py` in `DependencyParser` class: `compute_loss` method. Remember to implement the specific regularization mentioned in the paper.

Explain the high-level ideas of your implementations in the report.

## What experiments to try

You should try experiments to figure out the effects of following on learning:

1. activations (cubic vs tanh vs sigmoid)
2. pretrained embeddings
3. tunability of embeddings

Write your findings in the report.

As shown in the `experiments.sh`, you can use `--experiment-name` argument in the `train.py` to store the models at different locations in `serialization_dirs`. 
You can also use `--cache-processed-data` and `--use-cached-data` flags in `train.py` to not generate the training features everytime. Please look at training script for details. 
Lastly, after training your dev results will be stored in serialization directory of the experiment with name `metric.txt`.

You are supposed to submit all your models. Please name them appropriately (you can use the `experiment-name` argument). 
The model folders should be inside `serialization_dirs`. For your submission, rename it to `538_HW3_<SBUID>_models`. Zip it up and name the zip archive `538_HW3_<SBUID>_models.zip`. Upload this to google drive. You will sharing this link as part of your submission to Blackboard.

**NOTE**: You will be reporting the scores on development set but submitting the test prediction of the configuration that you found the best. The labels of test dataset are hidden from you.

## Runtimes and Outputs

The file `experiments.sh` lists the commands you will need to train and save these models. In all you will need ~5 full training runs.
I have tested the following on a Macbook Pro 2018 (no other processes like Zoom running).
Loading training instances from scratch takes ~8 minutes, while loading them from cache takes ~40s.
Each epoch takes ~4m (train + validation) on CPU, with the default settings.
With GloVe embeddings, it takes ~1m30s (train + validation) on CPU.
We have not tested this on Colab. We do not think you need a GPU with the above runtimes.

Using the default settings, the dev results are:
```
UAS: 84.73714385422639
UASnoPunc: 86.452269259029
LAS: 81.90293391828901
LASnoPunc: 83.28717571921099

UEM: 26.294117647058822
UEMnoPunc: 28.88235294117647
ROOT: 84.52941176470588
```
By adding the GloVe embeddings, they change to:
```
UAS: 85.12600643118877
UASnoPunc: 86.90725145537783
LAS: 82.45132986015903
LASnoPunc: 83.8834567343017

UEM: 28.470588235294116
UEMnoPunc: 30.529411764705884
ROOT: 85.41176470588235
```

Your implementation should have scores in the same ballpark with the default settings.
With your experiments, you are expected to improve upon these scores.

## Submission Instructions

Your submission folder should have the following structure:

```
538_HW3_<SBUID>
├── .gitignore
└── data/
  ├── train.conll
  ├── dev.conll
  ├── test.conll
  └── fixture.conll
└── lib/
  ├── __init__.py
  ├── configuration.py
  ├── data.py
  ├── dependency_tree.py
  ├── model.py
  ├── parsing_system.py
  ├── util.py
  └── fixture.conll
├── constants.py
├── download_glove.sh
├── evaluate.py
├── experiments.sh
├── predict.py
├── README.md
├── requirements.txt
└── submission/
  ├── gdrive_link.txt
  ├── Report_<SBUID>.pdf
  └── test_predictions.conll
└── train.py
```

Please keep the following things in mind:
1. Casing and spelling of file and folder names should be exactly the same as specified above.
2. We use underscores, not hyphens in filenames.
3. Zip `serialization_dirs` into `538_HW3_<SBUID>_models.zip` that will contain your 5 trained models (see `experiments.sh`). Upload this file to google drive and make the link public. Note that your google drive file should be called `538_HW3_<SBUID>_models`.
4. Your `gdrive_link.txt` file should contain just one line, with a `wget`-able link to the `538_HW3_<SBUID>_models` of your trained models.

### Good Luck!