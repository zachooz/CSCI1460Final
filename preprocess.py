from torch.utils.data import Dataset
import torch
import argparse
import os
from transformers import GPT2Tokenizer
from tqdm import tqdm


class SummaryDataset(Dataset):
    '''
        This processes the CNN DailyMail Dataset
        Sections are split by \n\n
        Sentence and labels are split by \t\t\t
        I chose to use the GPT2Tokenizer because it uses BPE trained on a large dataset
    '''
    def __init__(self, input_folder):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        files = os.listdir(input_folder)

        self.documents = []
        self.labels = []

        for file in tqdm(files):
            try:
                with open(input_folder + '/' + file) as f:
                    raw = f.read()
                    document, label = self.parse_document(raw)
                    self.documents.append(document)
                    self.labels.append(label)
            except Exception:
                print("failed to read", file)
                continue

    def parse_document(self, document):
        document = document.split('\n\n')
        document = self.insert_names(document[1], document[3])

        lines = []
        labels = []
        for line_label in document:
            line_label = line_label.split('\t\t\t')
            line = line_label[0]
            lines.append(self.tokenizer.encode(line, add_special_tokens=False))

            label = int(line_label[1])
            labels.append(1 if label == 1 else 0)
        return lines, labels

    def insert_names(self, document, names):
        '''
            This reinserts the names back into the texts. We can use the names without
            unking them because we are using GPT2's bpe
        '''
        parsed_doc = []
        document = document.split('\n')
        names = names.split('\n')

        entity_2_name = dict()
        for name in names:

            names = name.split(':')
            entity = names[0]
            name = ''.join(names[1:])
            entity_2_name[entity] = name

        for line in document:
            parsed_line = []
            for token in line.split(' '):
                if token in entity_2_name:
                    parsed_line.append(entity_2_name[token])
                else:
                    parsed_line.append(token)
            parsed_doc.append(' '.join(parsed_line))

        return parsed_doc


    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # TODO: Convert to tensor once we know what the model looks like
        item = {
            "document": self.documents[idx],
            "labels": self.labels[idx],
        }
        return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    train_dataset = SummaryDataset(args.folder)
