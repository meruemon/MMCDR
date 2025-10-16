# MMCDR

**Disentangling Shared and Specific Representations for Multimodal Cross-Domain Recommendation via Cosine-Based Regularization**  
*Submitted to IEICE Transactions on Information and Systems (IEICE Trans. Inf. & Sys.)*

## Overview

This repository contains the implementation of **MMCDR**, a novel framework for multimodal cross-domain recommendation. The proposed method disentangles shared and domain-specific representations using cosine-based regularization, enabling more effective knowledge transfer across domains and modalities.

## Requirements
- Python==3.7
- CUDA==11.6
- PyTorch==1.12.0
- tqdm==4.64.0
- torchsummary==1.5.1
- loguru
- sentence-transformer==2.2.2

## Datasets
We utilized the Amazon Reviews dataset. To download the Amazon dataset, you can use the following link: [Amazon Reviews](https://amazon-reviews-2023.github.io/). Download the four domains: Movies_and_TV, CDs_and_Vinyl, Sports_and_Outdoors, and Cell_Phones_and_Accessories. 

## Model Training

The model training process consists of two main steps: **Data Preprocessing** and **Model Training**.

---

### Step 1: Data Preprocessing

Before training the model, you need to preprocess the data and extract features from multiple modalities.

#### 1.1 Extract Common Users Across Domains
```shell
python data_preprocess.py
```
This script identifies and extracts users who appear in multiple domains.

#### 1.2 Extract Features from User Reviews
```shell
python process_review_feat.py
```
This script processes user review text and extracts textual features.

#### 1.3 Extract Features from Item Descriptions
```shell
python process_text_feat.py
```
This script extracts textual features from item descriptions.

#### 1.4 Extract Features from Item Images
```shell
# Download and organize image files
python process_img_file.py

# Extract visual features using pre-trained CNN
python process_visual.py
```
These scripts process item images and extract visual features.

---

### Step 2: Train the Model

Once all preprocessing is complete, you can train the MMCDR model:

```shell
python main.py --dataset phone_sport --batch_size 512 --lr 0.001 --epochs 200 --gamma 1 --embed_id_dim 128
```

**Note:** Ensure that all preprocessing steps are completed before running the training script.

## Results

Performance metrics evaluated with **Top-10 recommendations** across different domain pairs.

### Movie ↔ Music Domain Pair

|  Domain   |   HR   |  NDCG  |  MRR   | Precision |
|-----------|--------|--------|--------|-----------|
| **Movie** | 0.5801 | 0.3478 | 0.2762 |  0.0580   |
| **Music** | 0.6096 | 0.3656 | 0.2905 |  0.0610   |


### Sport ↔ Phone Domain Pair

|  Domain   |   HR   |  NDCG  |  MRR   | Precision |
|-----------|--------|--------|--------|-----------|
| **Sport** | 0.5146 | 0.2957 | 0.2289 |  0.0515   |
| **Phone** | 0.5228 | 0.3069 | 0.2409 |  0.0523   |

## Paper Status

The paper is currently **under review** at *IEICE Transactions on Information and Systems*.

**Note:**  
As the manuscript is under peer review, the repository is currently in a limited-release state. Some details, including datasets, trained models, and complete documentation, will be provided after the review process concludes.

## Citation

A BibTeX entry will be provided here upon acceptance.
