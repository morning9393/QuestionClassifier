import torch
from torch.utils.data import DataLoader
import numpy as np
import model as md


class QuestionClassifier:
    """
    Ensemble of based model, if ensemble size is 1, means a single model and will not apply ensemble strategy.

    :param ensemble_size: How many model in this ensemble. If 1, bootstrapping will not be applied.
    :param data_path: Data file path.
    :param vocabulary_path: Vocabulary file path.
    :param labels_path: Label file path.
    :param stop_words_path: Stop word file path.
    :param pre_train_path: Pre-trained embedding file path, None for random initialisation.
    :param k: k value, only words with frequency >= k will be reserved in vocabulary.
    """

    def __init__(self, ensemble_size, data_path, vocabulary_path, labels_path, stop_words_path, pre_train_path=None,
                 k=3):
        """Initialising, load data and bootstrapping to generate subset."""
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
        self.k = k
        self.load_dataset(data_path)
        self.init_subsets(ensemble_size)

    def load_dataset(self, path):
        """
        Load dataset from file, transfer to a list and assign to self.dataset.

        :param path: Data file path.
        """
        with open(path, 'r') as dataset_file:
            for line in dataset_file:
                label, question = line.split(' ', 1)
                self.dataset.append((label, question))

    def init_subsets(self, ensemble_size):
        """
        Use bootstrapping to generate a set of sub dataset, which used to build different models.

        :param ensemble_size: How many model in the committee. If 1, bootstrapping will not be applied.
        """
        if ensemble_size < 2:
            sub_dataset = md.QuestionSet(self.dataset, self.vocabulary_path, self.labels_path,
                                         self.stop_words_path, self.pre_train_path, self.k)
            self.subsets.append(sub_dataset)
        else:
            for i in range(0, ensemble_size):
                sample = self.bootstrapping()
                sub_dataset = md.QuestionSet(sample, self.vocabulary_path, self.labels_path, self.stop_words_path,
                                             self.pre_train_path, self.k)
                self.subsets.append(sub_dataset)

    def bootstrapping(self):
        """
        Generate a subset with randomly selecting N sample in dataset with replace.
        N is the size of dataset.

        :return: a subset with the same size of original dataset.
        """
        random_idx = np.random.choice(range(0, len(self.dataset)), len(self.dataset), replace=True)
        return [self.dataset[i] for i in random_idx]

    def train(self, model, embedding_dim, lstm_hidden, fc_input, fc_hidden, epochs, lr, freeze=True, test_path=None):
        """
        Train all the models in committee.

        :param model: Type of model, must be one of [bilstm/hybrid-cat/hybrid-add/cnn/bow].
        :param embedding_dim: Length of reserved vocabulary, used to build embedding layer.
        :param lstm_hidden: Dimension of lstm hidden state, used to build bilstm model.
        :param fc_input: Dimension of input full connect layer.
        :param fc_hidden: Dimension of hidden full connect layer.
        :param epochs: How many epochs it trains.
        :param lr: Learning rate.
        :param freeze: Freezing embedding layer or not. if True, weight in embedding layer will not be changed during training.
        :param test_path: Test file path, use to validate the model after every epoch.
        """
        self.model = model
        test_set = None
        loaders = []
        criterion = []
        optimizers = []
        if test_path is not None:
            test_set = md.QuestionSet(test_path, self.vocabulary_path, self.labels_path, self.stop_words_path,
                                      self.pre_train_path, self.k)
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
                print('ensemble: %d, model: %s, epoch: %d, stop_words: %s, pre_train_embedding: %s, k: %d, freeze: %s,'
                      ' train_set: %s, test_set: %s, acc: %d, acc rate: %f' %
                      (len(self.subsets), self.model, e + 1, self.stop_words_path, self.pre_train_path, self.k, freeze,
                       self.data_path, test_path, acc, acc_rate))

    def test(self, data_set, is_cnn=False, print_detail=False):
        """
        Validate this model with corresponding test set.

        :param data_set: Test set, QuestionSet type.
        :param is_cnn: Is this model based on cnn or not.
        :param print_detail: Print every sample and its result in the test set or not.
        :return: A tuple with hit number and accuracy rate.
        """
        data_loader = DataLoader(data_set)
        acc = 0
        for t, (cla, test) in enumerate(data_loader):
            vote = {}
            if self.model == 'cnn' or is_cnn:
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
        """
        Load models form path to build this ensemble.

        :param path: Model or sub-models path.
        """
        if self.ensemble_size < 2:
            net = torch.load(path)
            self.classifiers.append(net)
        else:
            for i in range(0, self.ensemble_size):
                i_path = path + '.' + str(i)
                net = torch.load(i_path)
                self.classifiers.append(net)

    def save(self, path):
        """
        Save every model in ensemble to path.

        :param path: Path to save.
        :return:
        """
        if len(self.classifiers) == 1:
            torch.save(self.classifiers[0], path)
        if len(self.classifiers) > 1:
            for i in range(0, len(self.classifiers)):
                i_path = path + '.' + str(i)
                torch.save(self.classifiers[i], i_path)

    @staticmethod
    def normalize(sample):
        """
        Normalize a list(question) of indexes to a unified length 21.
        Used when model is cnn.

        :param sample: A list(question) of indexes.
        :return: Unified list with length 21.
        """
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
