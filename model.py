from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SummaryModel(nn.Module):
    def __init__(self, vocab_size, word_rnn_size, sentence_rnn_size, embedding_size, max_sentence_size):
        """
        The Model class for SummaRuNNer

        :param vocab_size: The number of unique tokens in the data
        :param  word_rnn_size: The size of hidden cells in the world level GRU
        :param sentence_rnn_size: The size of hidden cells in the sentence level GRU
        :param embedding_size: The dimension of embedding space
        :max_sentence_size: The maximum sentence size
        """
        super().__init__()

        # TODO: initialize the vocab_size, rnn_size, embedding_size
        self.num_layers = 1
        self.num_directions = 2
        self.vocab_size = vocab_size
        self.word_rnn_size = word_rnn_size
        self.sentence_rnn_size = sentence_rnn_size
        self.embedding_size = embedding_size
        self.max_sentence_size = max_sentence_size

        # TODO: initialize embeddings, LSTM, and linear layers
        self.embedding = torch.nn.Embedding(vocab_size, self.embedding_size)
        self.word_gru = torch.nn.GRU(self.embedding_size, word_rnn_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.sentence_gru = torch.nn.GRU(
            self.word_rnn_size * max_sentence_size, sentence_rnn_size,
            num_layers=self.num_layers,
            bidirectional=True, batch_first=True
        )

        self.AvgPool1 = torch.nn.AvgPool1d(2)
        self.AvgPool2 = torch.nn.AvgPool1d(2)

        self.linear = torch.nn.Linear(sentence_rnn_size, 1)
        self.sigmoid = nn.Sigmoid()

    def word_level_gru(self, paragraph, sentence_lengths):
        """
        Given a list of sentences, run the word level bidirectional RNN to compute the sentence level representations
        for each sentence. The sentence level representation for a sentence is obtained by computing the hidden state
        representations at each word position sequentially, based on the current word embeddings and the previous
        hidden state. These hidden states are then concatenated together and average pooled to form the sentence
        level representation.

        :param paragraph: A tensor (paragraph_size, sentence_size) of sentences. Each sentence is a list of token ids.
        :param sentence_lengths: A tensor (paragraph_size,) representing the actual lengths (no padding) of each sentence in the paragraph
        """
        # Word level GRU applied to each sentence in paragraph
        paragraph_size = paragraph.size()[0]

        embeddings = self.embedding(paragraph) # Should have shape (max paragraph size, max sentence size, embedding size)

        packed_seq = pack_padded_sequence(embeddings, sentence_lengths, batch_first=True, enforce_sorted=False)

        # (paragraph_size, sentences_len, num_directions * hidden_size)
        word_level_representations, _ = self.word_gru(packed_seq)
        word_level_representations, lens_unpacked = pad_packed_sequence(
            word_level_representations,
            batch_first=True,
            total_length=self.max_sentence_size
        )

        # Average pool results
        word_level_representations = self.AvgPool1(word_level_representations)

        # Concatenate results (paragraph_size, sentences_len * hidden_size)
        # Gives 1 representation per sentence to give to GRU
        word_level_representations = word_level_representations.view(paragraph_size, -1)

        return word_level_representations

    def sentence_level_gru(self, paragraphs, paragraph_lengths):
        """
        Given a list of sentence reperesentations, run the sentence level bidirectional RNN to compute the paragraph
        level representations.

        :param hidden_states: A tensor containing paragraphs (batch_size, paragraph_size, sentence_representation_size)
        :param sentence_lengths: A tensor (batch_size,) containing the actual lengths of each sentence in the paragraph
        """
        # (batch_size, paragraph_len, num_directions * hidden_size)
        out, _ = self.sentence_gru(paragraphs)
        return self.AvgPool1(out)



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

        # TODO: Parallelize the model
        # To parallelize model we need to pass all sentences into a single call word_level_gru
        # and group their embeddings by paragraph. Then pass that to sentence_level_gru

        # Compute word level representations over all sentences
        # Result should have shape (batch_size, paragraph_size, word_rnn_size)

        word_level_outputs = [self.word_level_gru(inputs[i], sentence_lengths[i]) for i in range(len(inputs))]
        word_level_outputs = torch.stack(word_level_outputs)

        # Parallelization attempt
        # inputs = inputs.view(-1, self.max_sentence_size)
        # sentence_lengths = sentence_lengths.view(-1)
        # word_level_outputs = self.word_level_gru(inputs, sentence_lengths)

        # have to recombine to paragraphs
        # index = 0
        # paragraphs = []
        # for length in paragraph_lengths:
        #     paragraphs.append(torch.narrow(word_level_outputs, 0, index, index + length))
        #     index += length
        #
        # word_level_outputs = torch.stack(paragraphs)

        # print(paragraphs.size())

        # Run sentence level bidirectional GRU over the resulting hidden states
        # Should have shape (batch_size, paragraph_size, sentence_rnn_size)
        sentence_level_outputs = self.sentence_level_gru(word_level_outputs, paragraph_lengths)

        # TODO: Apply non-linear transformation as described in paper to get representation of document

        # Add logistic layer to make binary decision
        l1 = self.linear(sentence_level_outputs).squeeze(2)
        return self.sigmoid(l1)
