import json
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import re

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = SentenceTransformer('sentence-transformers/sentence-t5-base', device=device)

def load_data(domain):
    # Load test data
    df=pd.read_csv(f'/exports/eddie/scratch/s2550585/diss/dataset3/dataset/{domain}/{domain}.test.csv')
    return df

def load_data_maps(domain, experiment_type):

    with open(f'/exports/eddie/scratch/s2550585/diss/dataset3/dataset/datamap_exp/{domain}.data_maps', 'r') as f:
        data_maps = json.load(f)

    for key, value in tqdm(data_maps['id2meta'].items()):
        title = f'Title: {value}'
        description = f'Description: {data_maps["id2desc"].get(key, "No description available")}'
        
        if experiment_type == 'T':
            text = title
        elif experiment_type == 'T+D':
            text = f'{title} {description}'

        new_value = f'<extra_id_1>{text}<extra_id_2>'
        tokenized_value = model.tokenizer.tokenize(new_value)
        
        if len(tokenized_value) <= 255:
            data_maps['id2meta'][key] = new_value
        else:
            diff = len(tokenized_value) - 255
            truncated_value = '<extra_id_1>'+ text[:-diff] + '<extra_id_2>' 
            data_maps['id2meta'][key] = truncated_value
    
    return data_maps

def get_results(data, data_maps, column='history_text'):
    queries = {}
    relevant_docs = {}
    for _, row in data.iterrows():
        user_id = str(data_maps['user2id'][row['user_id']])
        item_id = str(data_maps['item2id'][row['parent_asin']])
        if user_id not in queries:
            queries[user_id] = row[column]
            relevant_docs[user_id] = []
        relevant_docs[user_id].append(item_id)
    
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=data_maps['id2meta'],
        relevant_docs=relevant_docs,
        ndcg_at_k=[10, 50],
        precision_recall_at_k=[10, 50],
        show_progress_bar=True,
        batch_size=32,
    )
    results = ir_evaluator(model)
    return results

def experiment(experiment_type, data, domain):
    data_maps = load_data_maps(domain, experiment_type)
    results = get_results(data, data_maps)
    return results

def run_experiments(domain):
    print(f"Running experiments for {domain}")
    data = load_data(domain)
    
    results = {}
    for exp_type in ['T','T+D']:
        print(f"Running experiment: {exp_type}")
        results[exp_type] = experiment(exp_type, data, domain)
    
    return results

# Run experiments for all domains
domains = ['All_Beauty', 'Baby_Products', 'Video_Games',
           'Beauty_and_Personal_Care', 'Cell_Phones_and_Accessories',
           'Electronics', 'Health_and_Household', 'Movies_and_TV', 'Toys_and_Games']

all_results = {}

for domain in domains:
    all_results[domain] = run_experiments(domain)

# Save detailed results to a file
with open('item_text_representation_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nDetailed results saved to 'item_text_representation_results.json'")