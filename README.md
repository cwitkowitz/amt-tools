# Automatic Music Transcription (AMT) Tools
Implements a customizable machine learning pipeline for AMT in PyTorch. This framework abstracts various elements of the AMT task, such as dataset usage, data formatting, feature extraction, model usage, output formatting, training, evaluation, and inference.
This makes for easy modification and extension through inheritance.

The framework is a work-in-progress. Its development is ongoing to meet my evolving research needs.

# Installation
##### Standard (PyPI)
Recommended for standard/quick usage
```
pip install amt-tools
```

##### Cloning Repository
Recommended for running examples or making experimental changes.
```
git clone https://github.com/cwitkowitz/amt-tools
pip install -e amt-tools
```

# Usage
This repository can be used for many different purposes. Please see each subpackage for more information.

Additionally, several papers are implemented under the ```examples``` subdirectory in standalone scripts which utilize the framework.
