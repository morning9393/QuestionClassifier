import torch
import random
import numpy as np

embedding = {}
stop_words = []


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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bag_of_word(question):
    bag = set(word for word in question.split() if word not in stop_words)
    vector = torch.zeros(50)
    for word in bag:
        if word in embedding.keys():
            vector += torch.tensor(embedding[word])
        else:
            vector += torch.tensor(embedding['unk'])
    return vector / len(bag)


setup_seed(16)
load_embedding()
load_stop_words()
q = 'What is considered the costliest disaster the insurance industry has ever faced ? What'
print(bag_of_word(q))
