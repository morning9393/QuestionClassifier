import numpy as np


def load_data():
    with open('../data/dataset.txt', 'r') as data_file:
        return data_file.read().split('\n')


def load_stop_words():
    with open('../data/stop_words.txt', 'r') as stop_words_file:
        return stop_words_file.read().split()


def load_glove():
    with open('../data/glove.200d.small.txt', 'r') as glove_file:
        glove_dict = {}
        for line in glove_file:
            glove_word = line.split()
            key = glove_word.pop(0)
            glove_dict[key] = glove_word
        return glove_dict


def generate_labels():
    label_set = set([])
    for data in load_data():
        label = data.split(' ', 1)[0]
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
    questions = [s.split(' ', 1)[1].lower() for s in data]
    return questions


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


np.random.seed(16)
write_labels()
write_vocabulary()
