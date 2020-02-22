import re
import numpy as np


def load_data():
    with open('../data/dataset.txt', 'r') as data_file:
        return data_file.read().split('\n')


def load_stop_words():
    with open('../data/stop_words.txt', 'r') as stop_words_file:
        return stop_words_file.read().split()


def load_glove():
    with open('../data/glove.6B.50d.txt', 'r') as glove_file:
        glove_dict = {}
        for line in glove_file:
            glove_word = line.split()
            key = glove_word.pop(0)
            glove_dict[key] = glove_word
        return glove_dict


def split_data():
    data = load_data()
    indexes = np.random.randint(0, len(data), int(len(data) * 0.1))
    dev = [data[i] for i in indexes]
    train = [s for s in data if s not in dev]
    return train, dev


def write_train_dev():
    train, dev = split_data()
    with open('../data/train.txt', 'w') as train_file:
        train_file.write('\n'.join(train))
    with open('../data/dev.txt', 'w') as dev_file:
        dev_file.write('\n'.join(dev))


def generate_labels():
    label_set = set([])
    for data in load_data():
        pattern = re.compile(r'\w+:\w+\s')
        label = pattern.search(data).group().strip()
        label_set.add(label)
    labels = list(label_set)
    labels.sort()
    return labels


def write_labels():
    with open('../data/labels.txt', 'w') as labels_file:
        labels = generate_labels()
        labels_file.write('\n'.join(labels))


def generate_corpus():
    data = load_data()
    questions = [re.sub(r'\w+:\w+\s', '', s).lower() for s in data]
    return questions


def write_corpus():
    with open('../data/corpus.txt', 'w') as corpus_file:
        questions = generate_corpus()
        corpus_file.write('\n'.join(questions))


def generate_vocabulary():
    stop_words = load_stop_words()
    words = ' '.join(generate_corpus()).split()
    print(len(words))
    vocabulary = {}
    for word in words:
        if word in stop_words:
            continue
        if word in vocabulary.keys():
            vocabulary[word] += 1
        else:
            vocabulary[word] = 1
    vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
    return vocabulary


def write_vocabulary():
    with open('../data/vocabulary.txt', 'w') as vocabulary_file:
        vocabulary = generate_vocabulary()
        word_count = sum(vocabulary.values())
        print(word_count)
        vocabs_str = [("%s %d" % (key, value)) for key, value in vocabulary.items()]
        vocabulary_file.write('\n'.join(vocabs_str))


def generate_pre_train_embedding():
    pre_train_embedding = []
    unk = list(np.random.rand(50))
    unk = [str(round(v, 6)) for v in unk]
    unk.insert(0, '#unk#')
    pre_train_embedding.append(' '.join(unk))
    vocabulary = generate_vocabulary()
    glove = load_glove()
    for vocab in vocabulary.keys():
        if vocab in glove.keys() and vocabulary[vocab] > 2:  # k = 3
            vec = glove[vocab]
            vec.insert(0, vocab)
            pre_train_embedding.append(' '.join(vec))
    return pre_train_embedding


def write_pre_train_embedding():
    with open('../data/embedding_glove.txt', 'w') as embedding_file:
        pre_train_embedding = generate_pre_train_embedding()
        embedding_file.write('\n'.join(pre_train_embedding))


np.random.seed(16)
write_train_dev()
write_labels()
write_corpus()
write_vocabulary()
write_pre_train_embedding()
