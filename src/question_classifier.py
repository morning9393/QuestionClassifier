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


def standardize(reps):
    matrix = [rep[1] for rep in reps]
    matrix = np.array(matrix)
    mean = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0)
    matrix = (matrix - mean) / sigma
    reps_std = []
    for i in range(0, len(reps)):
        reps_std.append((reps[i][0], list(matrix[i])))
    return reps_std


def load_labels():
    with open('../data/labels.txt', 'r') as labels_file:
        return labels_file.read().split('\n')


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
        return label, question

    def __len__(self):
        return len(self.questions)


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(200, 64)
        self.fc2 = torch.nn.Linear(64, 50)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out


setup_seed(16)
TRAIN_REP_PATH = '../data/train_rep.txt'
DEV_REP_PATH = '../data/dev_rep.txt'

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
dataLoader = get_data_loader(TRAIN_REP_PATH, batch_size=1)

for e in range(0, 100):
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
