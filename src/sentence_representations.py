import re
import torch


def load_embedding():
    with open('../data/embedding_glove.txt', 'r') as embedding_file:
        embedding = {}
        for line in embedding_file:
            pair = line.split()
            key = pair[0]
            value = [float(x) for x in pair[1:]]
            embedding[key] = value
        return embedding


def load_stop_words():
    with open('../data/stop_words.txt', 'r') as stop_words_file:
        return stop_words_file.read().split()


def bag_of_word(question, embedding, stop_words):
    bag = set(word for word in question.split() if word not in stop_words)
    vector = torch.zeros(200)
    for word in bag:
        if word in embedding.keys():
            vector += torch.tensor(embedding[word])
        else:
            vector += torch.tensor(embedding['#unk#'])
    return vector / len(bag)


def load_data_to_vector(path):
    with open(path, 'r') as data_file:
        data = []
        embedding = load_embedding()
        stop_words = load_stop_words()
        for line in data_file:
            pattern = re.compile(r'\w+:\w+\s')
            label = pattern.search(line).group().strip()
            question = pattern.sub('', line, 1).lower()
            data.append((label, bag_of_word(question, embedding, stop_words)))
        return data


def load_train():
    return load_data_to_vector('../data/train.txt')


def load_dev():
    return load_data_to_vector('../data/dev.txt')


def write_train_rep():
    with open('../data/train_rep.txt', 'w') as train_rep_file:
        train = load_train()
        for label, rep in train:
            reps = [str(round(v.item(), 6)) for v in rep]
            train_rep_file.write('%s %s\n' % (label, ' '.join(reps)))


def write_dev_rep():
    with open('../data/dev_rep.txt', 'w') as dev_rep_file:
        dev = load_dev()
        for label, rep in dev:
            rep_str = [str(round(v.item(), 6)) for v in rep]
            dev_rep_file.write('%s %s\n' % (label, ' '.join(rep_str)))


write_train_rep()
write_dev_rep()
