from comet_ml import Experiment
from model import SummaryModel
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import argparse
import numpy as np
from tqdm import tqdm  # optional progress bar

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "rnn_size": 128,
    "embedding_size": 512,
    "num_epochs": 5,
    "batch_size": 20,
    "learning_rate": 0.0001
}

if __name__ == "__main__":
   

    # TODO: Make sure you modify the `.comet.config` file
    #experiment = Experiment(log_code=False)
    #experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    
    vocab_size = 1000

    model = SummaryModel(
        vocab_size,
        128,
        128,
        512
    ).to(device)

    input = torch.LongTensor(np.random.randint(0,999,(5,30,10))).to(device)
    paragraph_lengths = torch.Tensor(np.random.randint(1,30,(5,)))
    sentence_lengths = torch.Tensor(np.random.randint(1,10,(5,30)))
    output = model(input, paragraph_lengths, sentence_lengths)

