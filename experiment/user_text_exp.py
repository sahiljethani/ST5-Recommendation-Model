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

def clean_text(text):
    # Remove all extra_id tags
    cleaned = re.sub(r'<extra_id_\d+>', '', text)
    # Split by newlines and join with commas
    return ','.join(line.strip() for line in cleaned.split('\n') if line.strip())

def load_and_preprocess_data(domain):
    # Load test data
    data = pd.read_csv(f'/kaggle/input/recdata/data/{domain}/{domain}.test.csv')
    
    # Preprocess history_text
    data['his_nostruc'] = data['history_text'].apply(clean_text)
    
    # Create structured and non-structured instruction columns
    instruction = "A user has purchased a sequence of items ordered in chronological order. Each item in the sequence is represented as \"Title: <item title>\" . The following sentence represents the user history: "
    data['inst_hist_struc'] = instruction + data['history_text']
    data['inst_hist_nostruc'] = instruction + data['his_nostruc']
    
    return data

def load_data_maps(domain):
    with open(f'/kaggle/input/recdata/data/{domain}/{domain}.data_maps', 'r') as f:
        return json.load(f)

def create_datamap(instruct, data_maps, tokenizer):
    if instruct == 'struct':
        for key, value in tqdm(data_maps['id2meta'].items()):
            value = 'Title:' + value
            new_value = '<extra_id_1>' + value + '<extra_id_2>'
            tokenized_value = tokenizer.tokenize(new_value)
            if len(tokenized_value) <= 255:
                data_maps['id2meta'][key] = new_value
            else:
                diff = len(tokenized_value) - 255
                new_value = '<extra_id_1>' + value[:-diff] + '<extra_id_2>'
                data_maps['id2meta'][key] = new_value
    else:
        for key, value in tqdm(data_maps['id2meta'].items()):
            new_value = 'Title:' + value 
            data_maps['id2meta'][key] = new_value
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
        ndcg_at_k=[1, 10, 50],
        precision_recall_at_k=[1, 10, 50],
        show_progress_bar=True,
        batch_size=32,
    )
    results = ir_evaluator(model)
    return results

def experiment(number, data, data_maps, tokenizer):
    if number == 1: #structured history_text
        data_maps = create_datamap('struct', data_maps, tokenizer)
        results = get_results(data, data_maps, column='history_text')
    elif number == 2: #unstructured history_text
        data_maps = create_datamap('not_struct', data_maps, tokenizer)
        results = get_results(data, data_maps, column='his_nostruc')
    elif number == 3: #structured instruction
        data_maps = create_datamap('struct', data_maps, tokenizer)
        results = get_results(data, data_maps, column='inst_hist_struc')
    elif number == 4: #unstructured instruction
        data_maps = create_datamap('not_struct', data_maps, tokenizer)
        results = get_results(data, data_maps, column='inst_hist_nostruc')
    return results

def run_experiments(domain):
    print(f"Running experiments for {domain}")
    data = load_and_preprocess_data(domain)
    data_maps = load_data_maps(domain)
    
    results = {}
    for exp_num in range(1, 5):
        print(f"Running experiment {exp_num}")
        results[exp_num] = experiment(exp_num, data, data_maps, model.tokenizer)
    
    return results

# Run experiments for all domains
domains = ['All_Beauty', 'Baby_Products', 'Video_Games',
           'Beauty_and_Personal_Care', 'Cell_Phones_and_Accessories',
           'Electronics', 'Health_and_Household', 'Movies_and_TV', 'Toys_and_Games']
           
all_results = {}

for domain in domains:
    all_results[domain] = run_experiments(domain)


# # Save detailed results to a file
with open('user_text_experiment_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
