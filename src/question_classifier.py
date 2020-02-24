import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class QuestionSet(Dataset):

    def __init__(self, data_path, vocabulary_path, labels_path, stop_words_path, pre_train_path=None):
        self.dataset = []
        self.labels = []
        self.vocabulary = []
        self.stop_words = []
        self.pre_weight = []
        self.pre_train_words = {}
        self.pre_train_path = pre_train_path
        self.load_dataset(data_path)
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
                sample = line.split(' ', 1)
                label = sample[0]
                question = sample[1]
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

    def label2index(self, label):
        return self.labels.index(label)

    def index2label(self, index):
        return self.labels[index.item()]

    def get_pre_train_weight(self):
        return torch.FloatTensor(self.pre_weight)


class Net(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim, label_size, model='bow', pre_train_weight=None, freeze=True):
        super(Net, self).__init__()
        self.model = model
        self.hidden_state_size = int(embedding_dim / 2)
        if pre_train_weight is None:
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            self.embeddingBag = torch.nn.EmbeddingBag(vocab_size, embedding_dim)
        else:
            self.embedding = torch.nn.Embedding.from_pretrained(pre_train_weight, freeze=freeze)
            self.embeddingBag = torch.nn.EmbeddingBag.from_pretrained(pre_train_weight, freeze=freeze)
        self.bilstm = torch.nn.LSTM(embedding_dim, self.hidden_state_size, bidirectional=True)
        self.fc1 = torch.nn.Linear(embedding_dim, 64)
        self.fc2 = torch.nn.Linear(64, label_size)

    def forward(self, x):
        if self.model == 'bilstm':
            embeds = self.embedding(x)
            seq_len = len(x[0])
            bilitm_out, _ = self.bilstm(embeds.view(seq_len, 1, -1))
            out = torch.cat((bilitm_out[0, 0, self.hidden_state_size:],
                             bilitm_out[seq_len - 1, 0, :self.hidden_state_size])).view(1, -1)
        else:  # default: bag of word
            out = self.embeddingBag(x)
        out = self.fc1(out)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out


# --------------- train ----------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(16)
TRAIN_PATH = '../data/train.txt'
DEV_PATH = '../data/dev.txt'
VOCABULARY_PATH = '../data/vocabulary.txt'
LABELS_PATH = '../data/labels.txt'
STOP_WORDS_PATH = '../data/stop_words.txt'
PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
EMBEDDING_DIM = 200

trainSet = QuestionSet(TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
dataLoader = DataLoader(trainSet)
net = Net(trainSet.vocab_size(), EMBEDDING_DIM, trainSet.label_size(), model='bilstm',
          pre_train_weight=trainSet.get_pre_train_weight(), freeze=False)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for e in range(0, 30):
    error = 0
    for t, (cla, train) in enumerate(dataLoader):
        optimizer.zero_grad()
        cla_pred = net(train)
        loss = criterion(cla_pred, cla)
        error += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('%d epoch finish, loss: %f' % (e + 1, error / dataLoader.__len__()))
torch.save(net, '../data/model.bin')

# --------------- train set test----------------
acc = 0
for t, (cla, test) in enumerate(dataLoader):
    output = net(test)
    _, pred = torch.max(output.data, 1)
    if cla == pred:
        acc += 1
print('train set acc: ' + str(acc))
acc_rate = float(acc) / float(dataLoader.__len__())
print('train set acc_rate: ' + str(acc_rate))

# --------------- dev set test----------------
devSet = QuestionSet(DEV_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
devDataLoader = DataLoader(devSet)
acc = 0
for t, (cla, test) in enumerate(devDataLoader):
    output = net(test)
    _, pred = torch.max(output.data, 1)
    print('y: %s, pred: %s ' % (devSet.index2label(cla), devSet.index2label(pred)))
    if cla == pred:
        acc += 1
print('dev set acc: ' + str(acc))
acc_rate = float(acc) / float(devDataLoader.__len__())
print('dev set acc_rate: ' + str(acc_rate))
