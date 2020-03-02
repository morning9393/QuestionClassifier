import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import model as md


class QuestionClassifier:

    def __init__(self, ensemble_size, data_path, vocabulary_path, labels_path, stop_words_path, pre_train_path=None):
        self.model = None
        self.dataset = []
        self.subsets = []
        self.classifiers = []
        self.ensemble_size = ensemble_size
        self.data_path = data_path
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

    def train(self, model, embedding_dim, lstm_hidden, fc_input, fc_hidden, epochs, lr, freeze=True, test_path=None):
        self.model = model
        test_set = None
        loaders = []
        criterion = []
        optimizers = []
        if test_path is not None:
            test_set = md.QuestionSet(test_path, self.vocabulary_path, self.labels_path, self.stop_words_path,
                                      self.pre_train_path)
        for i in range(0, len(self.subsets)):
            loaders.append(DataLoader(self.subsets[i]))
            net = md.Net(model, self.subsets[i].vocab_size(), embedding_dim, lstm_hidden, fc_input, fc_hidden,
                         self.subsets[i].label_size(), pre_train_weight=self.subsets[i].get_pre_train_weight(),
                         freeze=freeze)
            self.classifiers.append(net)
            criterion.append(torch.nn.CrossEntropyLoss())
            optimizers.append(torch.optim.SGD(net.parameters(), lr=lr))

        for e in range(0, epochs):
            for i in range(0, len(self.subsets)):
                for t, (cla, train) in enumerate(loaders[i]):
                    if self.model == 'cnn':
                        train = self.normalize(train)
                    self.classifiers[i].train()
                    optimizers[i].zero_grad()
                    cla_pred = self.classifiers[i](train)
                    loss = criterion[i](cla_pred, cla)
                    loss.backward()
                    optimizers[i].step()
            if test_set is not None:
                acc, acc_rate = self.test(test_set)
                print(
                    'ensemble: %d, model: %s, epoch: %d, pre_train_embedding: %s, freeze: %s, train_set: %s, '
                    'test_set: %s, acc: %d, acc rate: %f' %
                    (len(self.subsets), self.model, e + 1, self.pre_train_path, freeze, self.data_path, test_path, acc,
                     acc_rate))

    def test(self, data_set, print_detail=False):
        data_loader = DataLoader(data_set)
        acc = 0
        for t, (cla, test) in enumerate(data_loader):
            vote = {}
            if self.model == 'cnn':
                test = self.normalize(test)
            for net in self.classifiers:
                net.eval()
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
                question = data_set.index2question(t)
                print('%s %s -> %s ' % (y, question, voted_pred))
        acc_rate = float(acc) / float(data_loader.__len__())
        return acc, acc_rate

    def load(self, path):
        if self.ensemble_size < 2:
            net = torch.load(path)
            self.classifiers.append(net)
        else:
            for i in range(0, self.ensemble_size):
                i_path = path + '.' + str(i)
                net = torch.load(i_path)
                self.classifiers.append(net)

    def save(self, path):
        if len(self.classifiers) == 1:
            torch.save(self.classifiers[0], path)
        if len(self.classifiers) > 1:
            for i in range(0, len(self.classifiers)):
                i_path = path + '.' + str(i)
                torch.save(self.classifiers[i], i_path)

    @staticmethod
    def normalize(sample):
        length = 21
        if len(sample[0]) < length:
            temp1 = np.zeros(length - len(sample[0]))
            temp2 = sample.numpy()
            temp3 = np.append(temp2, temp1)
            result = torch.LongTensor([temp3])
            return result
        else:
            result = sample[:, 0:length]
            return result


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
    TEST_PATH = '../data/test.txt'
    VOCABULARY_PATH = '../data/vocabulary.txt'
    LABELS_PATH = '../data/labels.txt'
    STOP_WORDS_PATH = '../data/stop_words.txt'
    PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
    ENSEMBLE_SIZE = 1  # the best 20
    MODEL = 'cnn'  # the best hybrid-cat
    EMBEDDING_DIM = 200
    LSTM_HIDDEN = 100  # the best 100
    FC_INPUT = 784  # the best 200 / 400 for cat / 784 for cnn
    FC_HIDDEN = 64  # the best 64
    EPOCHS = 20  # the best 30
    LEARNING_RATE = 0.01  # the best 0.01
    FREEZE = False  # the best False

    classifier = QuestionClassifier(ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH,
                                    PRE_TRAIN_PATH)
    classifier.train(MODEL, EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH)


# run()
# single
# best accuracy = 0.746

# ensemble
# 25 30  accuracy = 0.787
# 20 30  accuracy = 0.776
# 15 30  accuracy = 0.774
# 10 30  accuracy = 0.782
