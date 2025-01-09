# Leveraging Sentence-T5 for Sequential Recommendation

## Introduction

This project reformulates sequential recommendation as a sentence retrieval task, leveraging Large Language Models (LLMs). We convert user sequences and item descriptions into sentences, then utilize the Sentence-T5 base model (ST5) to encode them into sentence embeddings. These embeddings are then used to match user sequence embeddings to the item corpus for retrieving recommended items.

Link to the paper: https://www.researchgate.net/publication/387509183_Leveraging_Sentence-T5_for_Sequential_Recommendation

## Project Structure

### dataset/
- `data_preprocessing.py`: Preprocesses domains from Amazon Review'23 dataset
- `augmenting.py`: Contains augmentation functions for user data
- `yelp_dataset.ipynb`: Notebook for creating and preprocessing the Yelp 2018 dataset

### st5_training/
- `pretrain_item.py`: Item-Description contrastive pretraining for ST5 model
- `pretrain_seq.py`: Sequence-Sequence contrastive pretraining for ST5 model
- `seq_item.py`: Sequence-Item contrastive finetuning for ST5 model

### unisrec_sasrec_training/
- `train_unisrec.py`: Script for training UniSRec model
- `train_sasrec.py`: Script for training SASRec model

### Root directory
- `metric.py`: Implementation of NDCG and HR metrics
- `popularity.ipynb`: Notebook for popularity-based baseline method

### experiment/
Contains scripts for various experiments:
- User text and item text representation strategies
- Performance variation with user history length for various models
- Rating prediction task
- Quality of representation experiments


## Requirements

To install the required libraries, run the following commands:

For ST5 training (sentence-transformers library):
```
pip install -U sentence-transformers
```

For UniSRec and SASRec (using RecBole library):
```
pip install recbole

pip install kmeans_pytorch
```





