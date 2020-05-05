from torch.utils.data import Dataset
import torch
import argparse
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer


class SummaryDataset(Dataset):
    '''
        This processes the CNN DailyMail Dataset
        Sections are split by \n\n
        Sentence and labels are split by \t\t\t
        I chose to use the GPT2Tokenizer because it uses BPE trained on a large dataset
    '''
    def __init__(self, input_folder, tokenizer, max_sentence_size):
        self.tokenizer = tokenizer
        self.documents = []
        self.labels = []
        self.lengths = []
        self.max_len = 0
        # 200 threshold has a 97% retention rate and prevents misparsed documents
        # with 200+ line words from getting added to the dataset
        self.max_sentence_size = max_sentence_size

        files = os.listdir(input_folder)
        for file in tqdm(files):
            try:
                with open(input_folder + '/' + file) as f:
                    raw = f.read()
                    document, label, lengths, skipped = self.parse_document(raw)

                    if len(document) < 1 or skipped > 0.1:
                        continue
                    self.documents.append(pad_sequence(document, batch_first=True, max_len=self.max_sentence_size))
                    self.labels.append(label)
                    self.lengths.append(lengths)
                # print(len(self.documents))
            except Exception as e:
                # print("failed to read", file, e)
                continue

    def parse_document(self, document):
        document = document.split('\n\n')
        # link = document[0]
        document = self.insert_names(document[1], document[3])

        lines = []
        lengths = []
        labels = []
        max_len = 0
        skipped = 0
        for line_label in document:
            line_label = line_label.split('\t\t\t')
            line = line_label[0]
            encoded = self.tokenizer.encode(line, add_special_tokens=False)
            if len(encoded) > self.max_sentence_size:
                skipped += 1
                continue
            lengths.append(len(encoded))
            max_len = len(encoded) if len(encoded) > max_len else max_len
            lines.append(torch.tensor(encoded))
            label = int(line_label[1])
            labels.append(0.5 if label == 2 else label)
        return lines, torch.tensor(labels), torch.tensor(lengths), skipped / len(document)

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
            "paragraph_length": self.documents[idx].size()[0],
            "sentence_length": self.lengths[idx],
            "document": self.documents[idx],
            "labels": self.labels[idx],
        }
        return item


def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=None):
    r"""Pad a list of variable length Tensors with ``padding_value``

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.
        max_len (int): value to pad sequence to (T)

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


if __name__ == "__main__":
    # python preprocess.py neuralsum/dailymail/test
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    train_dataset = SummaryDataset(args.folder, tokenizer, 75)
