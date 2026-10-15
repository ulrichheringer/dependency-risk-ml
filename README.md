# Dependency Update Risk Estimation

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  [![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)  [![MIT License](https://img.shields.io/badge/License-MIT-2ea44f?style=for-the-badge)](LICENSE)  ![Synthetic Data](https://img.shields.io/badge/Data-Synthetic-6b7280?style=for-the-badge)

Estimates regression risk in dependency updates from version changes, dependency impact, and test-coverage signals.

## Data

Deterministic synthetic data is generated locally in `pipeline.py`. No private data or external download is required.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Evaluation results

With seed 42, a stratified holdout of 400 synthetic updates reached 0.8975 accuracy, 0.8844 precision, 0.8793 recall, 0.8818 F1, and 0.9706 ROC AUC. The target label is generated from a probabilistic rule over the same synthetic signals, so these scores do not measure risk for real packages.

## Limitations

The examples and risk labels are artificial. Real use would require dependency-update history with reliable outcomes and validation on updates that were not used for training.
