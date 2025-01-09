from tqdm import tqdm
import numpy as np
import os
import pandas as pd 
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import torch
import argparse


device=('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('sentence-transformers/sentence-t5-base', device=device) #change path to the model you want to evaluate
# List of domains
domains = ['All_Beauty', 'Baby_Products','Video_Games',
        'Beauty_and_Personal_Care','Cell_Phones_and_Accessories',
        'Electronics','Health_and_Household','Movies_and_TV','Toys_and_Games']


base_dir='/kaggle/input/recdata/data'

tokenizer = model.tokenizer

# Dictionary to store results for all domains
all_results = {}

for domain in domains:
    print(f"Processing domain: {domain}")

    #Load Data Maps
    data_maps_file = os.path.join(base_dir, domain, f"{domain}.data_maps")

    # Load test data
    df=pd.read_csv(os.path.join(base_dir, domain, f"{domain}.test.csv"))

    # Load data maps
    with open(data_maps_file, 'r') as f:
        data_maps = json.load(f)
        
    for key,value in tqdm(data_maps['id2meta'].items()):
        new_value='<extra_id_1>'+value+'<extra_id_2>'
        #tokenize it
        tokenized_value=tokenizer.tokenize(new_value)
        if len(tokenized_value)<=255:
            data_maps['id2meta'][key]=new_value
        else:
            diff=len(tokenized_value)-255
            new_value='<extra_id_1>'+value[:-diff]+'<extra_id_2>'
            data_maps['id2meta'][key]=new_value
    

    # Prepare queries and relevant_docs
    queries = {}
    relevant_docs = {}
    for _, row in df.iterrows():
        user_id = str(data_maps['user2id'][row['user_id']])
        item_id = str(data_maps['item2id'][row['parent_asin']])

        if user_id not in queries:
            queries[user_id] = row['history_text']
            relevant_docs[user_id] = []

        relevant_docs[user_id].append(item_id)

    #improve datamaps
    tokenizer = model.tokenizer


    # Create and run evaluator
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=data_maps['id2meta'],
        relevant_docs=relevant_docs,
        ndcg_at_k=[10,50],
        precision_recall_at_k=[10,50],
        show_progress_bar=True,
        batch_size=32,
        write_csv=True
    )
    results = ir_evaluator(model)
    print(f"{domain}: NDCG@10:{results['cosine_ndcg@10']:.4f} NDGC@50:{results['cosine_ndcg@50']:.4f} HR@10:{results['cosine_recall@10']:.4f} HR@50:{results['cosine_recall@50']:.4f}")
