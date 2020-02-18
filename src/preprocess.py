import re
import numpy as np

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
              'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
              'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
              'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
              'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
              'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
              'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
              'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
              'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
              "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
              'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
              "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
              "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
              "wouldn't", ",", "?", ".", "'s", "'t", "n't"]

np.random.seed(16)
with open('../data/dataset', 'r') as dataset:
    data = dataset.read()
    data_list = data.split('\n')

with open('../data/dev', 'w') as dev:
    dev_index = np.random.randint(0, len(data_list), int(len(data_list) * 0.1))
    dev_set = [data_list[i] for i in dev_index]
    dev.write('\n'.join(dev_set))

with open('../data/train', 'w') as train:
    train_set = [s for s in data_list if s not in dev_set]
    train_data = '\n'.join(train_set)
    train.write(train_data)

with open('../data/corpus', 'w') as corpus:
    train_data = re.sub(r'\w+:\w+\s', '', train_data)
    train_data = train_data.lower()
    corpus.write(train_data)

with open('../data/dictionary', 'w') as dictionary:
    train_list = train_data.split()
    print(len(train_list))
    train_dict = {}
    for d in train_list:
        if d in stop_words:
            continue
        if d in train_dict:
            train_dict[d] += 1
        else:
            train_dict[d] = 1
    train_dict = dict(sorted(train_dict.items(), key=lambda x: x[1], reverse=True))

    word_count = sum(train_dict.values())
    print(word_count)
    cover = 0
    vocabs = []
    for key, value in train_dict.items():
        if value > 2:  # k = 3, cover rate = 74.3%
            dictionary.write("%s %d\n" % (key, value))
            vocabs.append(key)
            cover += value
    print("cover: %d" % cover)

unk = list(np.random.rand(50))
unk = [str(round(v, 6)) for v in unk]
unk.insert(0, 'unk')
with open('../data/random_embedding', 'w') as random_embd:
    random_embd.write(' '.join(unk) + '\n')
    for vocab in vocabs:
        vec = list(np.random.rand(50))
        vec = [str(round(v, 6)) for v in vec]
        vec.insert(0, vocab)
        random_embd.write(' '.join(vec) + '\n')

with open('../data/glove.6B.50d.txt', 'r') as glove:
    glove_dict = {}
    for line in glove:
        glove_word = line.split()
        key = glove_word.pop(0)
        glove_dict[key] = glove_word

with open('../data/glove_embedding', 'w') as glove_embd:
    glove_embd.write(' '.join(unk) + '\n')
    for vocab in vocabs:
        if vocab in glove_dict.keys():
            vec = glove_dict[vocab]
            vec.insert(0, vocab)
            glove_embd.write(' '.join(vec) + '\n')
