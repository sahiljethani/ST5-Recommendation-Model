import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer
from utils import get_model, create_dataset
import os 
import json



def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_single(model_name, domain, base_dir, pretrained_file, **kwargs):

    # configurations initialization
    props = ['overall.yaml', f'{model_name}.yaml']
    kwargs['checkpoint_dir'] = os.path.join(base_dir, domain, "checkpoint_unisrec")
    print(props)

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    print(config)

    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model_class(config, train_data.dataset).to(device)

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    print("Best valid result: ", best_valid_result)
    print("Test result: ", test_result)
    print("Best valid score: ", best_valid_score)

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UniSRec', help='model name')
    parser.add_argument('--dataset', type=str, default='All_Beauty', help='dataset name')
    parser.add_argument('--base_dir', type=str, default='/kaggle/working/', help='Base directory for the dataset')
    parser.add_argument('--pretrained_file', type=str, default='', help='Pre-trained model path')
    args, unparsed = parser.parse_known_args()
    print(args)

    run_single(args.model,args.dataset,args.base_dir,args.pretrained_file)




