import re
import torch

embedding = {}
stop_words = []
train_set = []


def load_embedding():
    with open('../data/glove_embedding', 'r') as glove:
        global embedding
        for line in glove:
            pair = line.split()
            key = pair[0]
            value = [float(x) for x in pair[1:]]
            embedding[key] = value


def load_stop_words():
    with open('../data/stop_words', 'r') as stop:
        global stop_words
        stop_words = stop.read().split()


def load_train():
    with open('../data/train', 'r') as train:
        global train_set
        for line in train:
            pattern = re.compile(r'\w+:\w+\s')
            label = pattern.search(line).group().strip()
            question = pattern.sub('', line, 1).lower()
            train_set.append((label, bag_of_word(question)))


def write_sentence_rep():
    with open('../data/sentence_rep', 'w') as sentence_rep:
        for label, rep in train_set:
            rep_str = [str(round(v.item(), 6)) for v in rep]
            sentence_rep.write('%s %s\n' % (label, ' '.join(rep_str)))


def bag_of_word(question):
    bag = set(word for word in question.split() if word not in stop_words)
    vector = torch.zeros(50)
    for word in bag:
        if word in embedding.keys():
            vector += torch.tensor(embedding[word])
        else:
            vector += torch.tensor(embedding['unk'])
    return vector / len(bag)


load_embedding()
load_stop_words()
load_train()
write_sentence_rep()
