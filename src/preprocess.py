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

with open('../data/train', 'r') as train:
    data = train.read()

with open('../data/corpus', 'w') as corpus:
    data = re.sub(r'\w+:\w+\s', '', data)
    data = data.lower()
    corpus.write(data)

with open('../data/dictionary', 'w') as dictionary:
    data_list = data.split()
    print(len(data_list))

    data_dict = {}
    for d in data_list:
        if d in stop_words:
            continue
        if d in data_dict:
            data_dict[d] += 1
        else:
            data_dict[d] = 1
    data_dict = dict(sorted(data_dict.items(), key=lambda x: x[1], reverse=True))

    word_count = sum(data_dict.values())
    print(word_count)
    cover = 0
    vocabs = []
    for key, value in data_dict.items():
        if value > 2:  # k = 3, cover rate = 75.6%
            dictionary.write("%s %d\n" % (key, value))
            vocabs.append(key)
            cover += value
    print("cover: %d" % cover)

np.random.seed(16)
with open('../data/random_embedding', 'w') as embedding:
    for vocab in vocabs:
        vec = list(np.random.rand(30))
        vec = [str(v) for v in vec]
        vec.insert(0, vocab)
        embedding.write(' '.join(vec) + '\n')
