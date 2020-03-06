import torch
from torch.utils.data import Dataset
import numpy as np


class QuestionSet(Dataset):
    """
    Subclass of Dataset, load dataset, vocabulary, labels, stop words and pre-trained embedding.

    :param data: Dataset, could be a path or a list.
    :param vocabulary_path: Vocabulary file path.
    :param labels_path: Label file path.
    :param stop_words_path: Stop word file path.
    :param pre_train_path: Pre-trained embedding file path, None for random initialisation.
    :param k: k value, only words with frequency >= k will be reserved in vocabulary.
    """

    def __init__(self, data, vocabulary_path, labels_path, stop_words_path, pre_train_path=None, k=3):
        """Initialise QuestionSet"""
        self.dataset = []
        self.labels = []
        self.vocabulary = []
        self.stop_words = []
        self.pre_weight = []
        self.pre_train_words = {}
        self.pre_train_path = pre_train_path
        self.k = k
        if type(data) == str:
            self.load_dataset(data)
        else:
            self.dataset = data
        self.load_labels(labels_path)
        self.load_vocabulary(vocabulary_path)
        self.load_stop_words(stop_words_path)

    def __getitem__(self, index):
        """
        Get a sample according to its index.

        :param index: The position of a sample in dataset.
        :return: A tuple with label and question sentence.
        """
        label, question = self.dataset[index]
        label = self.label2index(label)
        question = self.question2indexes(question)
        question = torch.LongTensor(question)
        return label, question

    def __len__(self):
        """
        Get the length of dataset.

        :return: An int value indicating the length of dataset.
        """
        return len(self.dataset)

    def load_dataset(self, path):
        """
        Load dataset from file, transfer to a list and assign to self.dataset.

        :param path: Data file path.
        """
        with open(path, 'r') as dataset_file:
            for line in dataset_file:
                label, question = line.split(' ', 1)
                question = question.strip()
                self.dataset.append((label, question))

    def load_labels(self, path):
        """
        Load labels from file, transfer to a list and assign to self.labels.

        :param path: Label file path.
        """
        with open(path, 'r') as labels_file:
            self.labels = labels_file.read().split('\n')

    def load_vocabulary(self, path):
        """
        Load vocabulary from file, filter out words with frequency less than 3,
        transfer to a list and assign to self.vocabulary.
        If pre-trained embedding is set, only words in pre-trained embedding file will be reserved,
        pre-trained embedding weight is set at the same time.
        All other words are replace with '#unk#'.

        :param path: Vocabulary file path.
        """
        with open(path, 'r') as vocabulary_file:
            self.vocabulary.append('#unk#')
            if self.pre_train_path is None:
                for line in vocabulary_file:
                    pair = line.split()
                    if int(pair[1]) >= self.k:
                        self.vocabulary.append(pair[0])
            else:
                pre_train_dim = self.load_pre_train(self.pre_train_path)
                self.pre_weight.append(np.random.rand(pre_train_dim))
                for line in vocabulary_file:
                    pair = line.split()
                    if int(pair[1]) >= self.k and pair[0] in self.pre_train_words.keys():
                        self.vocabulary.append(pair[0])
                        self.pre_weight.append(self.pre_train_words[pair[0]])

    def load_pre_train(self, path):
        """
        Load pre-trained embedding from file, transfer to a dictionary and assign to self.pre_train_words.
        Key is the word and Value is corresponding vector.

        :param path: Pre-trained embedding file path.
        :return: Dimension of pre-trained word vector.
        """
        with open(path, 'r') as pre_train_file:
            pre_train_dim = 0
            for line in pre_train_file:
                pair = line.split()
                key = pair[0]
                value = [float(x) for x in pair[1:]]
                pre_train_dim = len(value)
                self.pre_train_words[key] = value
            return pre_train_dim

    def load_stop_words(self, path):
        """
        Load stop words from file, transfer to a list and assign to self.stop_words.

        :param path: Stop word file path.
        """
        with open(path, 'r') as stop_words_file:
            self.stop_words = stop_words_file.read().split()

    def vocab_size(self):
        """
        Get the length of reserved vocabulary.

        :return: An int value indicating the length of reserved vocabulary.
        """
        return len(self.vocabulary)

    def label_size(self):
        """
        Get the length of label set.

        :return: An int value indicating the length of label set.
        """
        return len(self.labels)

    def question2indexes(self, question):
        """
        Transfer a question sentence to a list with indexes of words.
        Each index corresponds to a word in vocabulary, used to map word to its embedding.

        :param question: Question sentence.
        :return: A list with indexes of every word in the question.
        """
        question = question.lower()
        indexes = []
        for word in question.split():
            if word in self.stop_words:
                continue
            if word in self.vocabulary:
                indexes.append(self.vocabulary.index(word))
            else:
                indexes.append(self.vocabulary.index('#unk#'))
        return indexes

    def index2question(self, index):
        """
        Find the question sentence according its position(index).

        :param index: Position of target question in dataset.
        :return: Target question sentence.
        """
        return self.dataset[index][1]

    def label2index(self, label):
        """
        Use index number to represent labels.

        :param label: Target label(string).
        :return: Position of target label in label set.
        """
        return self.labels.index(label)

    def index2label(self, index):
        """
        Find the label string according to its position(index).

        :param index: Position of target label in label set
        :return: Target label(string).
        """
        return self.labels[index.item()]

    def get_pre_train_weight(self):
        """
        Get pre-trained embedding weight which build in load_vocabulary().

        :return: A VxD tensor matrix, V is the number of word and D is the embedding dimension. None if it is not set.
        """
        if self.pre_train_path is None:
            return None
        else:
            return torch.FloatTensor(self.pre_weight)


class Net(torch.nn.Module):
    """
    Contain 5 based models: bilstm/hybrid-cat/hybrid-add/cnn/bow

    :param model: Type of model, must be one of [bilstm/hybrid-cat/hybrid-add/cnn/bow].
    :param vocab_size: Length of reserved vocabulary, used to build embedding layer.
    :param embedding_dim: Dimension of word embedding, used to build embedding layer.
    :param lstm_hidden: Dimension of lstm hidden state, used to build bilstm model.
    :param fc_input: Dimension of input full connect layer.
    :param fc_hidden: Dimension of hidden full connect layer.
    :param label_size: Dimension of output full connect layer. used as input to softmax layer.
    :param pre_train_weight: Pre-trained embedding weight, a VxD tensor matrix.
    :param freeze: Freezing embedding layer or not. if True, weight in embedding layer will not be changed during training.
    """

    def __init__(self, model, vocab_size, embedding_dim, lstm_hidden, fc_input, fc_hidden, label_size,
                 pre_train_weight=None, freeze=True):
        """Initialise model according to parameters."""
        super(Net, self).__init__()
        self.model = model
        self.lstm_hidden = lstm_hidden
        if pre_train_weight is None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.embeddingBag = torch.nn.EmbeddingBag(vocab_size, embedding_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
            self.embeddingBag = torch.nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze)
        self.conv1 = torch.nn.Conv2d(1, 1, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.bilstm = torch.nn.LSTM(embedding_dim, self.lstm_hidden, bidirectional=True)
        self.fc1 = torch.nn.Linear(fc_input, fc_hidden)
        self.fc2 = torch.nn.Linear(fc_hidden, label_size)

    def forward(self, x):
        """
        Feedforward process, according to model type to decide what to do.

        :param x: Input vector, a list of word indexes represents a question.
        :return: 50 dimension vector used as input to softmax layer in CrossEntropyLoss().
        """
        if self.model == 'bilstm':
            embeds = self.embedding(x)
            seq_len = len(x[0])
            bilitm_out, _ = self.bilstm(embeds.view(seq_len, 1, -1))
            out = torch.cat((bilitm_out[0, 0, self.lstm_hidden:],
                             bilitm_out[seq_len - 1, 0, :self.lstm_hidden])).view(1, -1)
        elif self.model == "hybrid-cat":
            embeds = self.embedding(x)
            seq_len = len(x[0])
            bilitm_out, _ = self.bilstm(embeds.view(seq_len, 1, -1))
            out_bilstm = torch.cat((bilitm_out[0, 0, self.lstm_hidden:],
                                    bilitm_out[seq_len - 1, 0, :self.lstm_hidden])).view(1, -1)
            out_bag = self.embeddingBag(x)
            out = torch.cat((out_bag, out_bilstm), 1)
        elif self.model == "hybrid-add":
            embeds = self.embedding(x)
            seq_len = len(x[0])
            bilitm_out, _ = self.bilstm(embeds.view(seq_len, 1, -1))
            out_bilstm = torch.cat((bilitm_out[0, 0, self.lstm_hidden:],
                                    bilitm_out[seq_len - 1, 0, :self.lstm_hidden])).view(1, -1)
            out_bag = self.embeddingBag(x)
            out = out_bag + out_bilstm
        elif self.model == "cnn":
            embeds = self.embedding(x)
            out = embeds.view(1, 1, len(embeds[0]), -1)
            out = self.conv1(out)
            out = torch.nn.functional.relu(out)
            out = self.pool(out)
            out = out.view(1, -1)
        else:  # default: bag of word
            out = self.embeddingBag(x)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out
