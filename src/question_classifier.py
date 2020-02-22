import re
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_questions(path):
    with open(path, 'r') as questions:
        reps = []
        for line in questions:
            pattern = re.compile(r'\w+:\w+\s')
            label = pattern.search(line).group().strip()
            vector = pattern.sub('', line, 1).split()
            vector = [float(element) for element in vector]
            reps.append((label, vector))
        return reps


def load_labels():
    with open('../data/labels.txt', 'r') as labels_file:
        return labels_file.read().split('\n')
        # full_labels = labels.read().split('\n')
        # super_label_set = set([])
        # for label in full_labels:
        #     super_label = to_super_label(label)
        #     super_label_set.add(super_label)
        # supper_label_list = list(super_label_set)
        # supper_label_list.sort()
        # return supper_label_list


def to_super_label(label):
    return label.split(':')[0]


def get_data_loader(data_path, batch_size):
    questions = QuestionSet(data_path)
    loader = DataLoader(questions, batch_size=batch_size, shuffle=True)
    return loader


class QuestionSet(Dataset):

    def __init__(self, data_path):
        self.questions = load_questions(data_path)
        self.labels = load_labels()

    def __getitem__(self, index):
        question = self.questions[index][1]
        question = torch.tensor(question, requires_grad=True)
        label = self.labels.index(self.questions[index][0])
        # label = self.questions[index][0]
        # label = self.labels.index(to_super_label(label))
        return label, question

    def __len__(self):
        return len(self.questions)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(50, 50)
        self.fc2 = torch.nn.Linear(50, 50)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        predicted = torch.nn.functional.softmax(out, dim=1)
        return predicted


setup_seed(16)
TRAIN_REP_PATH = '../data/train_rep.txt'
DEV_REP_PATH = '../data/dev_rep.txt'

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
dataLoader = get_data_loader(TRAIN_REP_PATH, batch_size=1)

for e in range(0, 100):
    for t, (cla, train) in enumerate(dataLoader):
        optimizer.zero_grad()
        cla_pred = net(train)
        loss = criterion(cla_pred, cla)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('%d epoch finish' % (e + 1))
torch.save(net, '../data/model.bin')

# --------------- train set test----------------
trainDataLoader = get_data_loader(TRAIN_REP_PATH, batch_size=1)
acc = 0
for t, (cla, test) in enumerate(trainDataLoader):
    output = net(test)
    _, pred = torch.max(output.data, 1)
    if cla == pred:
        acc += 1
print('train set acc: ' + str(acc))
acc_rate = float(acc) / float(trainDataLoader.__len__())
print('train set acc_rate: ' + str(acc_rate))

# --------------- dev set test----------------
devDataLoader = get_data_loader(DEV_REP_PATH, batch_size=1)
acc = 0
for t, (cla, test) in enumerate(devDataLoader):
    output = net(test)
    _, pred = torch.max(output.data, 1)
    print('y: %s, pred: %s ' % (cla, pred))
    if cla == pred:
        acc += 1
print('dev set acc: ' + str(acc))
acc_rate = float(acc) / float(devDataLoader.__len__())
print('dev set acc_rate: ' + str(acc_rate))
