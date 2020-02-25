import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from question_classifier import QuestionSet, Net


class EnsembleClassifier:
    EMBEDDING_DIM = 200
    HIDDEN_SIZE = 64

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
        for i in range(0, ensemble_size):
            sample = self.bootstrapping()
            sub_dataset = QuestionSet(sample, self.vocabulary_path, self.labels_path, self.stop_words_path,
                                      self.pre_train_path)
            self.subsets.append(sub_dataset)

    def bootstrapping(self):
        random_idx = np.random.choice(range(0, len(self.dataset)), len(self.dataset), replace=True)
        return [self.dataset[i] for i in random_idx]

    def train(self, epochs, lr, step_size, gamma):
        for subset in self.subsets:
            loader = DataLoader(subset)
            net = Net(subset.vocab_size(), self.EMBEDDING_DIM, self.HIDDEN_SIZE, subset.label_size(), model='bow',
                      pre_train_weight=subset.get_pre_train_weight(), freeze=True)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            print('%d classifier begin' % (self.subsets.index(subset) + 1))
            for e in range(0, epochs):
                error = 0
                for t, (cla, train) in enumerate(loader):
                    optimizer.zero_grad()
                    cla_pred = net(train)
                    loss = criterion(cla_pred, cla)
                    error += loss.item()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print('%d epoch finish, loss: %f' % (e + 1, error / loader.__len__()))

            self.classifiers.append(net)

    def test(self, data_set):
        data_loader = DataLoader(data_set)
        acc = 0
        for t, (cla, test) in enumerate(data_loader):
            vote = {}
            for net in self.classifiers:
                output = net(test)
                _, pred = torch.max(output.data, 1)
                pred = data_set.index2label(pred)
                y = data_set.index2label(cla)
                print('y: %s, sub classifier pred: %s ' % (y, pred))
                if pred in vote.keys():
                    vote[pred] += 1
                else:
                    vote[pred] = 1
            voted_pred = sorted(vote.items(), key=lambda x: x[1], reverse=True)[0][0]
            y = data_set.index2label(cla)
            if y == voted_pred:
                acc += 1
        print('acc: ' + str(acc))
        acc_rate = float(acc) / float(data_loader.__len__())
        print('acc_rate: ' + str(acc_rate))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run():
    TRAIN_PATH = '../data/train.txt'
    DEV_PATH = '../data/dev.txt'
    VOCABULARY_PATH = '../data/vocabulary.txt'
    LABELS_PATH = '../data/labels.txt'
    STOP_WORDS_PATH = '../data/stop_words.txt'
    PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
    ENSEMBLE_SIZE = 10
    EPOCHS = 10
    LEARNING_RATE = 0.1
    STEP_SIZE = 5
    GAMMA = 0.5

    setup_seed(16)
    classifer = EnsembleClassifier(ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH,
                                            PRE_TRAIN_PATH)
    classifer.train(EPOCHS, LEARNING_RATE, STEP_SIZE, GAMMA)
    test_set = QuestionSet(DEV_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    classifer.test(test_set)

run()