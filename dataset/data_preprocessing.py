import os
import re
import html
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings 
from transformers import AutoModel, AutoTokenizer
import numpy as np

warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='All_Beauty')
    parser.add_argument('--max_his_len', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='/exports/eddie/scratch/s2550585/diss/dataset3/dataset')
    parser.add_argument('--plm', type=str, default='sentence-transformers/sentence-t5-base')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--subset',type=bool,default=True)
    parser.add_argument('--samples',type=int,default=300000)
    return parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def filter_items_wo_metadata(example, item2meta):
    if example['parent_asin'] not in item2meta:
        example['history'] = ''
    history = example['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    example['history'] = ' '.join(filtered_history)
    return example


def truncate_text_history(example, max_his_len,item2meta,tokenizer,max_length=255):
    history_items=example['history'].split(' ')
    example['history'] = ' '.join(history_items[-max_his_len:])

    start_part = "<extra_id_0>" #Represent <seq_start>
    items=[f"<extra_id_1>Title:{item2meta[item]}<extra_id_2>" for item in history_items] #id 1 represent item start and id 2 represent item end
    end_part ="<extra_id_3>" #Represent <seq_end>  
    
    # Combine parts
    history_text = start_part + ''.join(items) + end_part
    tokens = tokenizer.tokenize(history_text)

    if len(tokens) <= max_length:
        example['history_text'] = history_text
        return example
    
    # Add items in reverse order while checking the length constraint
    final_items = []
    current_length = 2

    for item in reversed(items):
        new_length = current_length + len(tokenizer.tokenize(item))
        if new_length <= max_length:
            final_items.append(item)
            current_length = new_length
        else:
            break

    # Ensure the items are in the original order in the final result
    final_items.reverse()
    final_result = start_part + ''.join(final_items) + end_part

    example['history_text'] = final_result
    return example


def item_text(example,item2title,domain):
    if example['parent_asin'] not in item2title:
        example['target_item_text']=None
        return example
    target_item_text=item2title[example['parent_asin']]
    example['target_item_text']=target_item_text
    return example


def item_desc_text(example,item2description):
    if example['parent_asin'] not in item2description:
        example['target_description']=None  
        return example
    target_description=item2description[example['parent_asin']]
    example['target_description']=target_description
    return example

def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(dataset['user_id'], dataset['parent_asin'], dataset['history']):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item not in item2id:
                    item2id[item] = len(id2item)
                    id2item.append(item)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps

#cleaning preprocessing dataset

def list_to_str(l):
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l


def clean_text(raw_text):
    text = list_to_str(raw_text) #convert list to string
    text = html.unescape(text) #convert html entities to unicode
    text = text.strip() #remove leading and trailing whitespaces
    text = re.sub(r'<[^>]+>', '', text) #remove html tags
    text = re.sub(r'[\n\t]', ' ', text) #replace newlines and tabs with spaces
    text = re.sub(r' +', ' ', text) #replace multiple spaces with a single space
    text=re.sub(r'[^\x00-\x7F]', ' ', text) #remove non-ascii characters
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float): 
        sentence += str(feature) 
        sentence += '.' 
    elif isinstance(feature, list) and len(feature) > 0: 
        for v in feature: 
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2] 
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def clean_title(example):
    if 'title' in example and example['title']:
        return {'parent_asin': example['parent_asin'], 'title': feature_process(example['title'])}
    else:
        return {'parent_asin': example['parent_asin'], 'title': None}

def clean_description(example):
    if 'description' in example and example['description']:
        return {'parent_asin': example['parent_asin'], 'description': feature_process(example['description'])}
    else:
        return {'parent_asin': example['parent_asin'], 'description': None}



def process_sentence(sentence, max_length=255):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) <= max_length:
        return sentence

    flag = False

    while len(tokens) > max_length:
        start_index = sentence.find("<extra_id_1>")
        end_index = sentence.find("<extra_id_2>", start_index) + len("<extra_id_2>")

        if start_index == -1 or end_index == -1:
            raise AssertionError('Something wrong')

        sentence = sentence[:start_index] + sentence[end_index:]
        tokens = tokenizer.tokenize(sentence)

        if sentence.endswith('<extra_id_0><extra_id_3>'):
            flag = True
            break

    if flag:
        title_block = sentence[start_index:end_index]
        title_tokens = tokenizer.tokenize(title_block)
        title_token_ids = tokenizer.convert_tokens_to_ids(title_tokens)

        while len(tokens) > max_length - 2 and title_token_ids:
            title_token_ids.pop()
            truncated_title_block = tokenizer.decode(title_token_ids, skip_special_tokens=True)
            sentence = sentence[:start_index] + truncated_title_block + sentence[end_index:]
            tokens = tokenizer.tokenize(sentence)

        sentence += '<extra_id_2><extra_id_3>'

    return sentence


def process_meta(domain, n_workers):

    print(f'Processing metadata for {domain} domain')
    meta_dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        f'raw_meta_{domain}',
        split='full',
        trust_remote_code=True
    )

    title_dataset = meta_dataset.map(
        clean_title,
        num_proc=n_workers
    ).filter(lambda x: x['title'] is not None)

    description_dataset = meta_dataset.map(
        clean_description,
        num_proc=n_workers
    ).filter(lambda x: x['description'] is not None)

    item2title = {example['parent_asin']: example['title'] for example in tqdm(title_dataset)}
    item2description = {example['parent_asin']: example['description'] for example in tqdm(description_dataset)}

    return item2title,item2description
            

if __name__ == '__main__':
    args = parse_args()
    print(args)
    output_dir = os.path.join(args.output_dir, args.domain)
    check_path(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Processing {args.domain}...")
  
    item2title,item2description = process_meta(args.domain, args.n_workers)

    print(f"Processing {args.domain} interaction data...")
    
    datasets = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"0core_timestamp_w_his_{args.domain}",
        trust_remote_code=True
    )

    if args.subset:
        train_samples = args.samples
        valid_samples = int((2 / 7) * train_samples)
        test_samples = int((1 / 7) * train_samples)

        train_subset = datasets['train'].shuffle(seed=42).select(range(min(train_samples, len(datasets['train']))))
        valid_subset = datasets['valid'].shuffle(seed=42).select(range(min(valid_samples, len(datasets['valid']))))
        test_subset = datasets['test'].shuffle(seed=42).select(range(min(test_samples, len(datasets['test']))))

        datasets['train'] = train_subset
        datasets['valid'] = valid_subset
        datasets['test'] = test_subset


    # Load a pre-trained Sentence Transformer model
    model = SentenceTransformer('sentence-transformers/sentence-t5-base',device=device)

    # Access the tokenizer from the underlying transformers model
    tokenizer = model.tokenizer

    truncated_datasets = {}


    print("Processing interaction data...")

    for split in ['train','valid','test']:

        print(f"Processing {split} split...")
        filtered_dataset = datasets[split].map(
            lambda t: filter_items_wo_metadata(t, item2title),
            num_proc=args.n_workers
        )
        filtered_dataset = filtered_dataset.filter(lambda t: len(t['history']) > 0)

        truncated_dataset = filtered_dataset.map(
            lambda t: truncate_text_history(t, args.max_his_len, item2title,tokenizer),
            num_proc=args.n_workers
        )
        truncated_dataset = truncated_dataset.map(
            lambda t: item_text(t,item2title,args.domain),
            num_proc=args.n_workers
        )

        truncated_dataset = truncated_dataset.map(
            lambda t: item_desc_text(t,item2description),
            num_proc=args.n_workers
        )
        
        truncated_datasets[split] = truncated_dataset
        df=pd.DataFrame(truncated_datasets[split])
        df.to_csv(os.path.join(output_dir, f'{args.domain}.{split}.csv'), index=False)

        #For unisrec and sasrec: 
        df2 = df[['user_id','history','parent_asin','timestamp']]
        df2 = df2.rename(columns={'user_id':'user_id:token','history':'item_id_list:token_seq','parent_asin':'item_id:token','timestamp':'timestamp:float'})
        df2.to_csv(os.path.join(output_dir, f'{args.domain}.{split}.inter'), sep='\t', index=False)



    print("Remapping IDs...")
    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    id2desc = {0: '[PAD]'}
    for item in item2title:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2title[item]
        id2desc[item_id] = item2description[item]
    data_maps['id2meta'] = id2meta
    data_maps['id2desc'] = id2desc
    output_path = os.path.join(output_dir, f'{args.domain}.data_maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    print("Item Title")
    sorted_text = []    
    for i in range(1, len(data_maps['item2id'])):
        item_text=f"Category: {args.domain} Title: {data_maps['id2meta'][i]}"
        sorted_text.append(item_text)

    with open(os.path.join(output_dir, 'item_profile.txt'), 'w') as f:
        for line in sorted_text:
            f.write(f"{line}\n")
    
    '''
    Generate Item From Unisrec - roberta
    '''
    print("Generating item features for Unisrec (blair)")
    plm = 'hyp1231/blair-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(plm)
    model_plm = AutoModel.from_pretrained(plm).to(device)


    all_embeddings = []
    for pr in tqdm(range(0, len(sorted_text), args.batch_size)):
        batch = sorted_text[pr:pr + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model_plm(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings.tofile(os.path.join(output_dir, f'{args.domain}.{plm.split("/")[-1]}.feature'))

    print("Item features generated for Unisrec.")


    '''
    Statistics
    '''
    print("Statistics:")
    print(f'Domain: {args.domain}') 
    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")
    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            if len(history.split(' ')) == 1:
                n_interactions[split] += 1
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)






