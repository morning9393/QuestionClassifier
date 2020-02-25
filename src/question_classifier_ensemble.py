import torch
from torch.utils.data import DataLoader
import random
import numpy as np
from question_classifier import QuestionSet, Net


# --------------- train ----------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(16)
TRAIN_PATH = '../data/train_main.txt'
DEV_PATH = '../data/dev_main.txt'
VOCABULARY_PATH = '../data/vocabulary.txt'
LABELS_PATH = '../data/label_main.txt'
STOP_WORDS_PATH = '../data/stop_words.txt'
PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
EMBEDDING_DIM = 200
HIDDEN_SIZE = 8

trainSet = QuestionSet(TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
dataLoader = DataLoader(trainSet)
# net = Net(trainSet.vocab_size(), EMBEDDING_DIM, HIDDEN_SIZE, trainSet.label_size(), model='bow',
#           pre_train_weight=trainSet.get_pre_train_weight(), freeze=False)
net = Net(trainSet.vocab_size(), EMBEDDING_DIM, HIDDEN_SIZE, trainSet.label_size(), model='bow')
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

for e in range(0, 30):
    error = 0
    for t, (cla, train) in enumerate(dataLoader):
        optimizer.zero_grad()
        cla_pred = net(train)
        loss = criterion(cla_pred, cla)
        error += loss.item()
        loss.backward()
        optimizer.step()
    # scheduler.step()
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
