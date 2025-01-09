from tqdm import tqdm
import numpy as np
import os
import pandas as pd 
import os
import pandas as pd
import argparse
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import MultiDatasetBatchSamplers, SentenceTransformerTrainingArguments, BatchSamplers
from sentence_transformers.losses import Matryoshka2dLoss, MultipleNegativesSymmetricRankingLoss



os.environ['TRANSFORMERS_CACHE'] = '/exports/eddie/scratch/s2550585/huggingface_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = '/exports/eddie/scratch/s2550585/huggingface_cache/datasets'
os.environ['HF_HOME'] = '/exports/eddie/scratch/s2550585/huggingface_cache'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='data') #full data of nine domains
    parser.add_argument('--output_dir', type=str, default='dataset/')
    parser.add_argument('--plm', type=str, default='sentence-transformers/sentence-t5-base') #original st5 model
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--steps_eval', type=int, default=500)
    return parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(split, output_dir, domain,seed=36):
    print(f"Processing {split} split...")
    df = pd.read_csv(os.path.join(output_dir, f'{domain}.{split}.csv'))
    df = df.groupby('category', group_keys=False).apply(lambda x: x.sample(frac=1, random_state=36)) #shuffle withing category
    df = df.sample(frac=1, random_state=36).reset_index(drop=True) #shuffle entire dataset
    return df

def create_dataset(df, anchor_col, positive_col):
    return Dataset.from_pandas(df[[anchor_col, positive_col]].rename(columns={
        anchor_col: 'anchor',
        positive_col: 'positive'
    }))


if __name__ == '__main__':
    args = parse_args()
    print(args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    model = SentenceTransformer(args.plm, device=device)

    print(f"Dataset making for {args.domain}...")

    # Load datasets
    train_df = load_data("train", args.output_dir, args.domain)
    valid_df = load_data("valid", args.output_dir, args.domain)

    #Item-Desc Pretaining
    train_dataset=create_dataset(train_df, 'target_item_text', 'target_description')
    eval_dataset=create_dataset(valid_df[:70000], 'target_item_text', 'target_description')

    print("Datasets created.")

    # Define 
    mnrl_loss = losses.MultipleNegativesSymmetricRankingLoss(model=model)
  
    print("Losses defined.")

    model_path = os.path.join(args.output_dir, args.domain, 'pretrain-item')

    check_path(model_path)

    args = SentenceTransformerTrainingArguments(
        # Required parameter
        output_dir=model_path,

        # Training parameters
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="adamw_torch",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        weight_decay=0.01,

        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=args.steps,
        save_strategy="steps",
        save_steps=args.steps,
        save_total_limit=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # Logging and reporting
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=args.steps,
        report_to="none",  


        remove_unused_columns=False,  
        
    )


    print("Training the model...")

    # Define trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=mnrl_loss,
    )
    # Train the model
    trainer.train()

    print("Training complete.")

    # Save the best model
    best_model_dir = os.path.join(model_path, "best_model-pretrain-item")
    model.save(best_model_dir)
    print(f"Best model saved to {best_model_dir}")