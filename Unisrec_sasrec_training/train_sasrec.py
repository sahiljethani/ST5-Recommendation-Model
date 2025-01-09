import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer
from utils import get_model, create_dataset
import os
import json
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.data.dataloader import TrainDataLoader
from recbole.sampler import Sampler
from recbole.model.sequential_recommender import SASRec

#Function to filter out items that are not in the training set
def filter_data_by_valid_items(data, n_items,config): 
    item_seqs = data.inter_feat[config['ITEM_SEQ_FIELD']]
    mask = torch.all(item_seqs < n_items, dim=1)
    filtered_inter_feat = data.inter_feat[mask]
    filtered_data = data.copy(filtered_inter_feat)
    return filtered_data

def main(base_dir, domain):
    parameter_dict = {
        'train_neg_sample_args': None,
        'epochs': 300,
        'train_batch_size': 2048,
        'eval_batch_size': 2048,
        'learning_rate': 0.001,
        'embedding_size': 64,
        'hidden_size': 64,
        'num_layers': 2,
        'num_heads': 2,
        'dropout_prob': 0.2,
        'loss_type': 'CE',
        'metrics': ["Hit", "NDCG"],
        'topk': [10, 50],
        'valid_metric': 'NDCG@10',
        'data_path': base_dir,
        'dataset': domain,
        'benchmark_filename': ['train', 'valid', 'test'],
        'field_separator': "\t",
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'ITEM_SEQ_FIELD': 'item_id_list',
        'checkpoint_dir': os.path.join(base_dir, domain, 'saved_sasrec'),
        'load_col': {'inter': ['user_id', 'item_id_list', 'item_id', 'timestamp']}
    }

    model_class = SASRec
    
    # Load configuration
    config = Config(model=model_class, dataset=domain, config_dict=parameter_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    # Dataset filtering
    dataset = create_dataset(config)
    
    # Dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # Calculate the number of items in the dataset
    n_items = len(dataset.inter_feat[config['ITEM_ID_FIELD']].unique())
    print(f"Total number of items in the dataset: {n_items}")

    # Apply filtering to train, valid, and test datasets
    filtered_train_dataset = filter_data_by_valid_items(train_data.dataset, n_items,config)
    filtered_valid_dataset = filter_data_by_valid_items(valid_data.dataset, n_items,config)
    filtered_test_dataset = filter_data_by_valid_items(test_data.dataset, n_items,config)

    # Create appropriate data loaders
    train_sampler = Sampler(config, filtered_train_dataset)
    train_data = TrainDataLoader(config, filtered_train_dataset, train_sampler, shuffle=True)
    
    valid_sampler = Sampler(config, filtered_valid_dataset)
    valid_data = FullSortEvalDataLoader(config, filtered_valid_dataset, valid_sampler)
    
    test_sampler = Sampler(config, filtered_test_dataset)
    test_data = FullSortEvalDataLoader(config, filtered_test_dataset, test_sampler)

    # Model loading and initialization
    model = model_class(config, filtered_train_dataset).to(config['device'])
    
    # Trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # Model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    print(f"Best valid score: {best_valid_score}")
    print(f"Best valid result: {best_valid_result}")
    
    # Model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])
    print(f"Test result: {test_result}")

    # Save test result as JSON
    result_filename = f"{domain}.sas.result.json"
    result_path = os.path.join(base_dir, domain, result_filename)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(test_result, f, indent=4)
    print(f"Test result saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SASRec training')
    parser.add_argument('--base_dir', type=str,default='dataset/',help='Base directory for data')
    parser.add_argument('--domain', type=str,default='All_Beauty',help='Domain name')
    
    args = parser.parse_args()
    
    main(args.base_dir, args.domain)