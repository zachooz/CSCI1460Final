from comet_ml import Experiment
from model import SummaryModel
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.nn import functional as F
import torch
import argparse
import numpy as np
from preprocess import SummaryDataset, pad_sequence
from tqdm import tqdm  # optional progress bar
from transformers import GPT2Tokenizer
import nltk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: If we want more than a batch size of 1, we need to pad whole documents, so each document is the same size
# bc sizes of dataloader tensors must match except in dimension 0
hyperparams = {
    "rnn_size": 128,
    "embedding_size": 512,
    "num_epochs": 1,
    "batch_size": 1,
    "learning_rate": 0.0005,
    "max_sentence_size": 75,
}

# Get summary for ith example in a dataset
def generate_summary(idx, dataset, model, tokenizer):
    batch = dataset[idx]

    paragraph_length = batch["paragraph_length"]
    sentence_length = batch["sentence_length"]
    x = batch["document"]
    paragraph_lengths = torch.LongTensor(np.array([paragraph_length])[None,:]).to(device)
    sentence_lengths = torch.LongTensor(np.array(sentence_length)[None,:]).to(device)
    x = torch.LongTensor(np.array(x)[None,:]).to(device)

    y_pred = torch.squeeze(model(x, paragraph_lengths, sentence_lengths))

    print("ORIGINAL:\n")
    for i in range(paragraph_length):
        sentence = x[0,i]
        print(tokenizer.decode(sentence[0:sentence_length[i]].detach().cpu().numpy()) + "\n")

    print("-----------------------------------------------")
    print("SUMMARY:\n")
    for i in range(paragraph_length):
        sentence = x[0,i]
        if (y_pred[i] > 0.5):
            print(tokenizer.decode(sentence[0:sentence_length[i]].detach().cpu().numpy()) + " [CONFIDENCE: {}]\n".format(y_pred[i]))




def train(model, train_loader, hyperparams):
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), hyperparams['learning_rate'])

    model = model.train()
    for epoch in range(hyperparams['num_epochs']):
        for batch in tqdm(train_loader):
            paragraph_length = batch["paragraph_length"]
            sentence_length = batch["sentence_length"]
            x = batch["document"]
            y = batch['labels']

            paragraph_length = paragraph_length.to(device)
            sentence_length = sentence_length.to(device)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x, paragraph_length, sentence_length)
            loss = loss_fn(y_pred.float(), y.float())

            loss.backward()  # calculate gradients
            optimizer.step()  # update model weights

            print("loss:", loss.item())

        torch.save(model.state_dict(), './model.pt')

def test(model, test_loader):
    model = model.eval()
    loss_fn = nn.BCELoss()

    for batch in tqdm(test_loader):
        paragraph_length = batch["paragraph_length"]
        sentence_length = batch["sentence_length"]
        x = batch["document"]
        y = batch['labels']

        paragraph_length = paragraph_length.to(device)
        sentence_length = sentence_length.to(device)
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x, paragraph_length, sentence_length)
        loss = loss_fn(y_pred.float(), y.float())
        print("loss", loss.item())


def validate(model, test_loader):
    model = model.eval()
    loss_fn = nn.BCELoss()

    for batch in tqdm(test_loader):
        paragraph_length = batch["paragraph_length"]
        sentence_length = batch["sentence_length"]
        x = batch["document"]
        y = batch['labels']

        paragraph_length = paragraph_length.to(device)
        sentence_length = sentence_length.to(device)
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x, paragraph_length, sentence_length)
        loss = loss_fn(y_pred.float(), y.float())

        print("loss", loss.item())

def summarize_file(model, file, tokenizer, max_sentence_size):
    model = model.eval()
    with open(file) as f:
        raw = f.read().replace('“', '"').replace('”', '"')
        sentences = nltk.tokenize.sent_tokenize(raw)
        print(sentences)
        document, sentence_length = parse_file(sentences, tokenizer, max_sentence_size)

    paragraph_length = document.size()[0]

    sentence_length = sentence_length.unsqueeze(0).to(device)
    paragraph_length = torch.tensor(paragraph_length).unsqueeze(0).to(device)
    document = document.unsqueeze(0).to(device)

    y_pred = model(document, paragraph_length, sentence_length)
    y_pred = y_pred.detach().cpu()[0].numpy()

    summary = []

    for i in range(len(sentences)):
        if y_pred[i] > 0.7:
            summary.append(sentences[i])

    return summary, len(summary), len(sentences)



def parse_file(document, tokenizer, max_sentence_size):
    # link = document[0]
    lines = []
    lengths = []
    for line in document:
        encoded = tokenizer.encode(line, add_special_tokens=False)
        if len(encoded) > max_sentence_size or len(encoded) < 1:
            print("SKIP", line)
            continue
        lengths.append(len(encoded))
        lines.append(torch.tensor(encoded))
    return pad_sequence(lines, batch_first=True, max_len=max_sentence_size), torch.tensor(lengths)

# python main.py -l -T data/dailymail/train
# python main.py -l -t data/dailymail/test
# python main.py -l -F data/examples/apclimate.txt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true",
                        help="load model.pt")
    parser.add_argument("-s", "--save", action="store_true",
                        help="save model.pt")
    parser.add_argument("-T", "--train", action="store",
                        help="run training loop")
    parser.add_argument("-F", "--file", action="store",
                        help="run training loop")
    parser.add_argument("-V", "--validate", action="store",
                        help="run training loop")
    parser.add_argument("-t", "--test", action="store",
                        help="run testing loop")
    parser.add_argument("-S", "--summary", action="store_true",
                        help="generate summaries")
    args = parser.parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    print("GATHERING DATA")
    train_dataset = None
    if args.train:
        train_dataset = SummaryDataset(
            input_folder=args.train,
            tokenizer=tokenizer,
            max_sentence_size=hyperparams['max_sentence_size']
        )

    validate_dataset = None
    if args.validate:
        validate_dataset = SummaryDataset(
            input_folder=args.validate,
            tokenizer=tokenizer,
            max_sentence_size=hyperparams['max_sentence_size']
        )

    test_dataset = None
    if args.test:
        test_dataset = SummaryDataset(
            input_folder=args.test,
            tokenizer=tokenizer,
            max_sentence_size=hyperparams['max_sentence_size']
        )

    train_loader = None
    if args.train:
        train_loader = DataLoader(
            train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    validate_loader = None
    if args.validate:
        train_loader = DataLoader(
            validate_dataset, batch_size=hyperparams['batch_size'])

    test_loader = None
    if args.test:
        test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'])

    model = SummaryModel(
        tokenizer.vocab_size + 1,
        hyperparams['rnn_size'],
        hyperparams['rnn_size'],
        hyperparams['embedding_size'],
        hyperparams['max_sentence_size']
    ).to(device)



    if args.load:
        print("loading saved model...")
        model.load_state_dict(torch.load('./model.pt', map_location=device))

    if args.file:
        print("summarizing file...")
        summary, sum_len, doc_len = summarize_file(model, args.file, tokenizer, hyperparams['max_sentence_size'])
        print("Used", sum_len, "of", doc_len, "lines")
        print(' '.join(summary))

    if args.summary:
        print("summarizing test data...")
        generate_summary(0, test_dataset, model, tokenizer)

    if args.train:
        print("running training loop...")
        train(model, train_loader, hyperparams)
    if args.save:
        print("saving model...")
        torch.save(model.state_dict(), './model.pt')

    if args.validate:
        print("running training loop...")
        validate(model, validate_loader)

    if args.test:
        print("testing models...")
        test(model, test_loader)







    # TODO: Make sure you modify the `.comet.config` file
    # experiment = Experiment(log_code=False)
    # experiment.log_parameters(hyperparams)

    # TODO: Load dataset
    # Hint: Use random_split to split dataset into train and validate datasets
    #
    # vocab_size = 1000
    # paragraph_size = 30
    #
    #
    #
    # input = torch.LongTensor(np.random.randint(0, 999, (5, 100, 10))).to(device)
    #
    # # (batch_size,)
    # paragraph_lengths = torch.Tensor(np.random.randint(1, 100, (5,))).to(device)
    #
    # # (batch_size, paragraph_size)
    # sentence_lengths = torch.Tensor(np.random.randint(1, 10, (5, 100))).to(device)
    # output = model(input, paragraph_lengths, sentence_lengths)
    # print(output.shape)
