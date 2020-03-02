import torch
from torch.utils.data import Dataset
import numpy as np


class QuestionSet(Dataset):

    def __init__(self, data, vocabulary_path, labels_path, stop_words_path, pre_train_path=None):
        self.dataset = []
        self.labels = []
        self.vocabulary = []
        self.stop_words = []
        self.pre_weight = []
        self.pre_train_words = {}
        self.pre_train_path = pre_train_path
        if type(data) == str:
            self.load_dataset(data)
        else:
            self.dataset = data
        self.load_labels(labels_path)
        self.load_vocabulary(vocabulary_path)
        self.load_stop_words(stop_words_path)

    def __getitem__(self, index):
        label, question = self.dataset[index]
        label = self.label2index(label)
        question = self.question2indexes(question)
        question = torch.LongTensor(question)
        return label, question

    def __len__(self):
        return len(self.dataset)

    def load_dataset(self, path):
        with open(path, 'r') as dataset_file:
            for line in dataset_file:
                label, question = line.split(' ', 1)
                question = question.strip()
                self.dataset.append((label, question))

    def load_labels(self, path):
        with open(path, 'r') as labels_file:
            self.labels = labels_file.read().split('\n')

    def load_vocabulary(self, path):
        with open(path, 'r') as vocabulary_file:
            self.vocabulary.append('#unk#')
            if self.pre_train_path is None:
                for line in vocabulary_file:
                    pair = line.split()
                    if int(pair[1]) > 2:  # k = 3
                        self.vocabulary.append(pair[0])
            else:
                self.load_pre_train(self.pre_train_path)
                self.pre_weight.append(np.random.rand(200))
                for line in vocabulary_file:
                    pair = line.split()
                    if int(pair[1]) > 2 and pair[0] in self.pre_train_words.keys():  # k = 3
                        self.vocabulary.append(pair[0])
                        self.pre_weight.append(self.pre_train_words[pair[0]])

    def load_pre_train(self, path):
        with open(path, 'r') as pre_train_file:
            for line in pre_train_file:
                pair = line.split(' ')
                key = pair[0]
                value = [float(x) for x in pair[1:]]
                self.pre_train_words[key] = value

    def load_stop_words(self, path):
        with open(path, 'r') as stop_words_file:
            self.stop_words = stop_words_file.read().split()

    def vocab_size(self):
        return len(self.vocabulary)

    def label_size(self):
        return len(self.labels)

    def question2indexes(self, question):
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
        return self.dataset[index][1]

    def label2index(self, label):
        return self.labels.index(label)

    def index2label(self, index):
        return self.labels[index.item()]

    def get_pre_train_weight(self):
        if self.pre_train_path is None:
            return None
        else:
            return torch.FloatTensor(self.pre_weight)


class Net(torch.nn.Module):

    def __init__(self, model, vocab_size, embedding_dim, lstm_hidden, fc_input, fc_hidden, label_size,
                 pre_train_weight=None, freeze=True):
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
            out = embeds.view(1, 1, len(embeds[0]), 200)
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
