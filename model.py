from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reverse( tensor):
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    return tensor.index_select(0, idx)

class SummaryModel(nn.Module):
    def __init__(self, vocab_size, word_rnn_size, sentence_rnn_size, embedding_size):
        """
        The Model class for SummaRuNNer

        :param vocab_size: The number of unique tokens in the data
        :param  word_rnn_size: The size of hidden cells in the world level GRU
        :param sentence_rnn_size: The size of hidden cells in the sentence level GRU
        :param embedding_size: The dimension of embedding space
        """
        super().__init__()

        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.vocab_size = vocab_size
        self.word_rnn_size = word_rnn_size
        self.sentence_rnn_size = sentence_rnn_size
        self.embedding_size = embedding_size

        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_size)
        self.word_gru_reversed = torch.nn.GRU(self.embedding_size, word_rnn_size, num_layers=1)
        self.word_gru = torch.nn.GRU(self.embedding_size, word_rnn_size, num_layers=1)
        self.sentence_gru = torch.nn.GRU(self.word_rnn_size, sentence_rnn_size, num_layers=1)
        self.sentence_gru_reversed = torch.nn.GRU(self.word_rnn_size, sentence_rnn_size, num_layers=1)

        self.AvgPool1 = torch.nn.AvgPool1d(2)
        self.AvgPool2 = torch.nn.AvgPool1d(2)

        #self.linear = torch.nn.Linear()


    

    """
    Given a list of sentences, run the word level bidirectional RNN to compute the word level representations
    for each sentence.

    :param paragraph: A list of sentences. Each sentence is a list of token ids
    :param sentence_lengths: The actual lengths (no padding) of each sentence in the paragraph
    """
    def compute_word_level_representation(self, paragraph, sentence_lengths):
        # Word level GRU applied to each sentence in paragraph
        embeddings = self.embedding(paragraph) # Should have shape (max paragraph size, max sentence size, embedding size)
        reversed_embeddings = reverse(embeddings)
        reversed_lengths = reverse(sentence_lengths)

        packed_seq = pack_padded_sequence(embeddings, sentence_lengths, batch_first=True, enforce_sorted = False)
        packed_seq_reversed = pack_padded_sequence(reversed_embeddings, reversed_lengths, batch_first=True, enforce_sorted = False)
        _, out = self.word_gru(packed_seq)
        _, out_reversed = self.word_gru_reversed(packed_seq_reversed)

        # Concatenate the two results and average pool them
        word_level_representations = torch.cat((out, out_reversed), 2)
        pooled = self.AvgPool1(word_level_representations)
        return torch.squeeze(pooled)

    def compute_sentence_level_representation(self, hidden_states, paragraph_lengths):
        
        hidden_states_reversed = reverse(hidden_states)
        reversed_lengths = reverse(paragraph_lengths)

        packed_seq = pack_padded_sequence(hidden_states, paragraph_lengths, batch_first=True, enforce_sorted = False)
        packed_seq_reversed = pack_padded_sequence(hidden_states_reversed, reversed_lengths, batch_first=True, enforce_sorted = False)
        _, out = self.sentence_gru(packed_seq)
        _, out_reversed = self.sentence_gru_reversed(packed_seq_reversed)

        # Concatenate the two results and average pool them
        sentence_representations = torch.cat((out, out_reversed), 2)
        pooled = self.AvgPool1(sentence_representations)
        return torch.squeeze(pooled)



    def forward(self, inputs, paragraph_lengths, sentence_lengths):

        """
        Runs the forward pass of the model.

        :param inputs: word ids (tokens) of shape (batch_size, sentence_window_size, word_window_size)
        :param sentence_lengths: array of shape (batch_size, sentence_window_size) representing
         the actual lengths (no padding) of each sentence in the paragraph
        :param paragraph_lengths: an array of shape (batch_size) representing the number of sentences in the paragraph

        :return: the logits, a tensor of shape
                 (batch_size, window_size, vocab_size)
        """

        #TODO: Figure out how to parallelize the model

        # Compute word level representations over all sentences
        #word_level_outputs = self.compute_sentence_representation(inputs, sentence_lengths)
        word_level_outputs =[self.compute_word_level_representation(inputs[i], sentence_lengths[i]) for i in range(len(inputs))]
        word_level_outputs = torch.stack(word_level_outputs)

        # Run sentence level bidirectional GRU over the resulting hidden states
        sentence_level_outputs = self.compute_sentence_level_representation(word_level_outputs, paragraph_lengths)
        print(sentence_level_outputs.shape)

        # Concatenate the two results and average pool them

        # TODO: Apply non-linear transformation as described in paper to get representation of document


        # Add logistic layer to make binary decision