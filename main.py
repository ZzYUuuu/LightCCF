import torch
import logging
from datetime import datetime
import utility.parser as parser
import utility.tools as tools
import utility.data_loader as data_loader
import os


def load_param():
    args = parser.parse_args()
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print("\t device:" + str(device) + str(args.gpu))
    return args, device

def load_log(args, model_name):
    if not os.path.exists('log/' + model_name):
        os.mkdir('log/' + model_name)
    if not os.path.exists('log/' + model_name + '/' + args.dataset):
        os.mkdir('log/' + model_name + '/' + args.dataset)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if args.log == 'None':
        logfile = os.path.join('log/' + model_name + '/' + args.dataset, f'{timestamp}.log')
    else:
        logstr = str(args.log)
        logfile = os.path.join('log/' + model_name + '/' + args.dataset, f'{logstr}.log')
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info("-----" * 10)
    logger.info(args)
    return logger


def load_models():
    model_name = "LightCCF"

    import_str = 'from ' + model_name + ' import Trainer'
    imported_module = {}
    exec(import_str, imported_module)
    Trainer = imported_module['Trainer']

    return model_name, Trainer


def load_dataset(args):
    dataset = data_loader.Data(args)

    return dataset


if __name__ == '__main__':
    print('\t Light Contrastive Collaborative Filtering')
    print('-' * 80)

    print('\t Loading parameter file...')
    args, device = load_param()
    print('-' * 80)

    print('\t Loading model ...')
    model_name, Trainer = load_models(args)
    print('-' * 80)

    print('\t Loading logger...')
    logger = load_log(args, model_name)
    print('-' * 80)
    print('\t Loading dataset file...')
    dataset = load_dataset(args)
    print('-' * 80)
    print('\t Loading recommender and run...')
    recommender = Trainer(args, dataset, device, logger)
    if recommender is not None:
        recommender.train()
    else:
        logger.error('Not found recommender')
