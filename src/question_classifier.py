import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from src import model as md


class QuestionClassifier:

    def __init__(self, ensemble_size, data_path, vocabulary_path, labels_path, stop_words_path, pre_train_path=None):
        self.dataset = []
        self.subsets = []
        self.classifiers = []
        self.vocabulary_path = vocabulary_path
        self.labels_path = labels_path
        self.stop_words_path = stop_words_path
        self.pre_train_path = pre_train_path
        self.load_dataset(data_path)
        self.init_subsets(ensemble_size)

    def load_dataset(self, path):
        with open(path, 'r') as dataset_file:
            for line in dataset_file:
                label, question = line.split(' ', 1)
                self.dataset.append((label, question))

    def init_subsets(self, ensemble_size):
        if ensemble_size < 2:
            sub_dataset = md.QuestionSet(self.dataset, self.vocabulary_path, self.labels_path,
                                         self.stop_words_path, self.pre_train_path)
            self.subsets.append(sub_dataset)
        else:
            for i in range(0, ensemble_size):
                sample = self.bootstrapping()
                sub_dataset = md.QuestionSet(sample, self.vocabulary_path, self.labels_path, self.stop_words_path,
                                             self.pre_train_path)
                self.subsets.append(sub_dataset)

    def bootstrapping(self):
        random_idx = np.random.choice(range(0, len(self.dataset)), len(self.dataset), replace=True)
        return [self.dataset[i] for i in random_idx]

    def train(self, model, embedding_dim, lstm_hidden, fc_input, fc_hidden, epochs, lr, freeze=True):
        for subset in self.subsets:
            loader = DataLoader(subset)
            net = md.Net(model, subset.vocab_size(), embedding_dim, lstm_hidden, fc_input, fc_hidden,
                            subset.label_size(), pre_train_weight=subset.get_pre_train_weight(), freeze=freeze)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            print('%d classifier begin' % (self.subsets.index(subset) + 1))
            for e in range(0, epochs):
                error = 0
                for t, (cla, train) in enumerate(loader):
                    print(train)
                    if model == 'cnn':
                        length = 19
                        print(len(train[0]))
                        if len(train[0]) < length:
                            temp1 = np.zeros(length - len(train[0]))
                            temp2 = train.numpy()
                            temp3 = np.append(temp2, temp1)
                            train = torch.LongTensor([temp3])
                    print(train)
                    optimizer.zero_grad()
                    cla_pred = net(train)
                    print(cla_pred)
                    loss = criterion(cla_pred, cla)
                    error += loss.item()
                    loss.backward()
                    optimizer.step()
                print('%d epoch finish, loss: %f' % (e + 1, error / loader.__len__()))
            self.classifiers.append(net)

    def test(self, data_set, print_detail=False):
        data_loader = DataLoader(data_set)
        acc = 0
        for t, (cla, test) in enumerate(data_loader):
            vote = {}
            for net in self.classifiers:
                output = net(test)
                _, pred = torch.max(output.data, 1)
                pred = data_set.index2label(pred)
                if pred in vote.keys():
                    vote[pred] += 1
                else:
                    vote[pred] = 1
            voted_pred = sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0]
            y = data_set.index2label(cla)
            if y == voted_pred:
                acc += 1
            if print_detail:
                print('%s -> %s ' % (y, voted_pred))
        acc_rate = float(acc) / float(data_loader.__len__())
        return acc, acc_rate


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run():
    setup_seed(16)
    TRAIN_PATH = '../data/train.5000.txt'
    DEV_PATH = '../data/dev.txt'
    VOCABULARY_PATH = '../data/vocabulary.txt'
    LABELS_PATH = '../data/labels.txt'
    STOP_WORDS_PATH = '../data/stop_words.txt'
    PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
    ENSEMBLE_SIZE = 1  # the best 20
    MODEL = 'cnn'  # the best hybrid-cat
    EMBEDDING_DIM = 200
    LSTM_HIDDEN = 100  # the best 100
    FC_INPUT = 98  # the best 200 / 400 for cat
    FC_HIDDEN = 64  # the best 64
    EPOCHS = 20  # the best 30
    LEARNING_RATE = 0.02  # the best 0.02
    FREEZE = False  # the best False

    classifier = QuestionClassifier(ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH,
                                    PRE_TRAIN_PATH)
    classifier.train(MODEL, EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE)
    test_set = md.QuestionSet(DEV_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    acc, acc_rate = classifier.test(test_set)
    print('acc: ' + str(acc))
    print('acc_rate: ' + str(acc_rate))


run()
# single
# best accuracy = 0.746

# ensemble
# 25 30  accuracy = 0.787
# 20 30  accuracy = 0.776
# 15 30  accuracy = 0.774
# 10 30  accuracy = 0.782
