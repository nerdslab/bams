# Relax, it doesn't matter how you get there: A new self-supervised approach for multi-timescale behavior analysis (NeurIPS 2023)

![Architecture Overview](overview.png)

This is the official PyTorch implementation of BAMS in 'Relax, it doesn't matter how you 
get there: A new self-supervised approach for multi-timescale behavior analysis' 
(NeurIPS 2023).

This repo contains examples of training BAMS on the Multi Agent Behavior Challenge (MABe) 
datasets, which are benchmark datasets that provide a rich set of labels to evaluate the 
quality of the learned representations. The main scripts for these datasets will be 
`mouse_triplets.py` and `fruit_flies.py`.

BAMS is a general purpose self-supervised learning method for behavior analysis, and does
not require labels. To train BAMS on your own dataset, please refer to `custom_dataset.py`.

### Setup

Clone this repository:
```bash
git clone https://github.com/nerdslab/bams.git
```

To set up a Python virtual environment with the required dependencies, run:
```bash
python3 -m venv bams_env
source bams_env/bin/activate
pip install --upgrade pip
pip install -e .
```

## Mouse Triplet experiments
To see an example of training and evaluating BAMS end-to-end, please follow the steps below to test BAMS on the Mabe mouse triplets challenge.

### 1. Downloading the MABe data

The [MABe 2022](https://sites.google.com/view/computational-behavior/our-datasets/mabe2022-dataset)
trajectory is publically available.
Run the following script to download the MABe data (both mouse triplet and fruit fly datasets):
```bash
bash download_mabe.sh
```
### 2. Model training
To start training, run:
```bash
python3 mouse_triplets.py --job train
```

You can track the runs in Tensorboard:
```bash
tensorboard --logdir runs
```

To compute the learned representations and save them to a file, run:
```bash
python3 mouse_triplets.py --job compute_representations --ckpt_path ./bams-mouse-triplet-2023-12-04-14-42-44.pt
```
### 3. Linear evaluation 
For linear evaluation of the learned representations, we will use the public
MABe evaluator:
```bash
cd mabe-2022-public-evaluator/
python3 round1_evaluator.py --task mouse --submission ../bams-mouse-triplet-2023-11-30-17-49-59_submission.npy --labels ../data/mabe/mouse_triplets_test_labels.npy
```

## Custom datasets
If you'd like to test BAMS on your own dataset, please use `custom_dataset.py`. 

### 1. Model training
To train a model, you will need to simply load your `keypoints` object which should be of shape `(n_samples, seq_len, num_feats)`. 

**Note:**  If you have missing values, or need to pad your data to the same length, please use `np.nan`
as the missing value.

### 2. Extract the embeddings

```
embs, hoa_pred, byol_preds = model(input)
```
embs is a dict with embs['short_term'] and embs['long_term'] containing both the short-term and long-term latent embeddings
