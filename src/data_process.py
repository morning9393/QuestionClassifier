import numpy as np


def load_data():
    """
    Load dataset from file '../data/dataset.txt' and transfer to a list.

    :return: A list whose each element corresponds to a line in the file.
    """
    with open('../data/dataset.txt', 'r') as data_file:
        return data_file.read().split('\n')


def load_stop_words():
    """
    Load stop words from file '../data/stop_words.txt' and transfer to a list.

    :return: A list with stop words.
    """
    with open('../data/stop_words.txt', 'r') as stop_words_file:
        return stop_words_file.read().split()


def generate_labels():
    """
    Generate a label list according to the first word in each line of dataset.

    :return: A list with labels.
    """
    label_set = set([])
    for data in load_data():
        label = data.split(' ', 1)[0]
        label_set.add(label)
    labels = list(label_set)
    labels.sort()
    return labels


def write_labels():
    """
    Write labels to file '../data/labels.txt', each line is a label
    """
    with open('../data/labels.txt', 'w') as labels_file:
        labels = generate_labels()
        labels_file.write('\n'.join(labels))


def generate_corpus():
    """
    Generate corpus from dataset, remove label from every question.

    :return: A list with questions
    """
    data = load_data()
    questions = [s.split(' ', 1)[1].lower() for s in data]
    return questions


def generate_vocabulary():
    """
    Generate vocabulary from dataset and count their frequency.

    :return: A dictionary with words in dataset and its frequency except stop word.
    """
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
    """
    Write vocabulary to '../data/vocabulary.txt', each line contains a word and its frequency.
    """
    with open('../data/vocabulary.txt', 'w') as vocabulary_file:
        vocabulary = generate_vocabulary()
        word_count = sum(vocabulary.values())
        print(word_count)
        vocabs_str = [("%s %d" % (key, value)) for key, value in vocabulary.items()]
        vocabulary_file.write('\n'.join(vocabs_str))

if __name__ == "__main__":
    np.random.seed(16)
    write_labels()
    write_vocabulary()
