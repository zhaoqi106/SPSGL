# SPSGL: Structural-Prior Guided Synaptic Graph Learning

---
SPSGL is a brain-graph learning framework designed for fMRI data, aiming to predict mental health–related outcomes.

## Environment

---

1.Create a new conda environment with python version 3.8.18.

```
conda create -n "spsgl" python=3.8.18
conda activate spsgl
```
2.Install PyTorch and torchvision.
```
We recommend an evironment with pytorch==2.0.0+cu118, torchvision==0.15.1+cu118, torchaudio==2.0.1+cu118
```
3.Other packages can be found in the requirements.txt file.

## Usage

1. Update the relevant file settings in /source/data_processing/data_build.py, then run the script.
2. Update path, label_columns, label_threshold, and label_sign in /source/conf/dataset/hcp.yaml to specify the dataset path, the target task labels, and the threshold settings.
3. Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model=spsgl dataset=hcp preprocess=mixup
```

