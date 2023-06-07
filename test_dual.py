import argparse
import time
import torch
import random
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from data_loader import DualGraphDataset
from model import get_model
from utils import print_metrics, labels_to_weights
from experiment import get_experiment
from transformers import logging
import warnings
import json

os.environ["CUDA_VISIBLE_DEVICES"]="0"

warnings.filterwarnings("ignore")

RANDOM_SEED = 1

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

logging.set_verbosity_error()

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(
    description='Dual Graph Rumor Detection and Verification (baseline)')

parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 16)')

parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=5e-5, metavar='N',
                    help='learning rate (default: 2e-5)')

parser.add_argument('--weight_decay', type=float, default=1e-2, metavar='N',
                    help='weight decay (default: 1e-3)')
                    
parser.add_argument('--hidden_dim', type=int, default=768, metavar='N',
                    help='hidden dimension (default: 768)')

parser.add_argument('--max_len', type=int, default=32, metavar='N',
                    help='maximum length of the conversation (default: 32)')

parser.add_argument('--dropout', type=float, default=0.1, metavar='N',
                    help='dropout rate (default: 0.1)')

parser.add_argument('--experiment', type=str, metavar='N',
                    help='experiment name')

parser.add_argument('--model', type=str, default="dual", metavar='N',
                    help='model name')

parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='experiment name')
parser.add_argument('--aug', type=bool, default=False, metavar='N',
                    help='experiment name')

args = parser.parse_args()


def eval():

    experiment = get_experiment(args.experiment)

    image_dir = experiment["image_dir"]

    root_dir = os.path.join(experiment["root_dir"], str(args.fold))

    language = experiment["language"]

    classes = experiment["classes"]

    test_dataset = DualGraphDataset(
        root=root_dir,
        image_dir=image_dir,
        classes=classes,
        split='test',
        language=language,
        max_length=args.max_len,
        aug=args.aug
        )    
        
    test_loader = DataListLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    print('testing samples : {} '.format(len(test_dataset)))

    model = get_model(args.model,args.hidden_dim, len(classes),
                         args.dropout, language=language)

    model = DataParallel(model).to(device)

    comment = f'dual_{args.model}_{args.experiment}_{args.fold}'

    result_dir = os.path.join("results/",comment)
    os.makedirs(result_dir,exist_ok=True)

    # writer = SummaryWriter(log_dir="runs/{}_{}".format(str(int(time.time())),"eval_" + comment))

    checkpoint_dir = os.path.join("checkpoints/",comment)

    with torch.no_grad():
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
        model.module.load_state_dict(torch.load(checkpoint_path))

    model.eval()

    test_loss = 0.0
    test_count = 0

    predicts = []
    test_labels = []
    results = []

    for _, batch in enumerate(test_loader):

        labels = torch.cat([data.y for data in batch]).to(device).long()

        outputs = model(batch)

        _, predict = torch.max(outputs, 1)

        id = str(batch[0].id)
        predict = int(predict.cpu().detach().numpy()[0])
        label = int(labels.cpu().detach().numpy()[0])
        results.append({"id": id, "label": label, "predict": predict})

        test_labels.append(label)
        predicts.append(predict)


    result_path = os.path.join(result_dir,"results.json")
    with open(result_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    eval()