import re
import torch
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_sentences():
    with open('../data/train_sentence_rep', 'r') as sentences:
        reps = []
        for line in sentences:
            pattern = re.compile(r'\w+:\w+\s')
            label = pattern.search(line).group().strip()
            vector = pattern.sub('', line, 1).split()
            vector = [float(x) for x in vector]
            reps.append((label, torch.tensor([vector], requires_grad=True)))
        return reps


def load_labels():
    with open('../data/labels', 'r') as labels:
        label_map = {}
        target = 0
        for line in labels:
            label_map[line.strip()] = torch.tensor([target], dtype=torch.long)
            target += 1
        return label_map


class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(50, 50)
        self.fc2 = torch.nn.Linear(50, 50)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = torch.nn.functional.relu(hidden)
        out = self.fc2(relu)
        predicted = torch.nn.functional.softmax(out, dim=1)
        return predicted


setup_seed(16)
reps = load_sentences()
targets = load_labels()

net = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

net.train()
for e in range(0, 2):
    for t in range(0, len(reps)):
        optimizer.zero_grad()

        x = reps[t][1]
        y = targets[reps[t][0]]
        y_pred = net(x)
        loss = criterion(y_pred, y)
        # print(loss)

        loss.backward()
        optimizer.step()
    scheduler.step()
    print('%d epoch finish' % (e + 1))
torch.save(net, '../data/model.bin')

net.eval()
acc = 0
for t in range(0, len(reps)):
    x = reps[t][1]
    y = targets[reps[t][0]]
    output = net(x)
    _, pred = torch.max(output.data, 1)
    print('y: %s, pred: %s ' % (y, pred))
    if y == pred:
        acc += 1
print('acc: ' + str(acc))
acc_rate = float(acc) / float(len(reps))
print('acc_rate: ' + str(acc_rate))

