import sys
import torch
import random
import time
import numpy as np
import question_classifier as cf
import model as md

TRAIN_PATH = '../data/train.5000.txt'
DEV_PATH = '../data/dev.txt'
TEST_PATH = '../data/test.txt'
VOCABULARY_PATH = '../data/vocabulary.txt'
LABELS_PATH = '../data/labels.txt'
STOP_WORDS_PATH = '../data/stop_words.txt'
PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
ENSEMBLE_SIZE = 1  # the best 20
MODEL = 'hybrid-cat'  # the best hybrid-cat
EMBEDDING_DIM = 200  # the best 200
LSTM_HIDDEN = 100  # the best 100
FC_INPUT = 200  # the best 200 / 400 for hybrid-cat / 784 for cnn
FC_HIDDEN = 64  # the best 64
EPOCHS = 1  # the best 30
LEARNING_RATE = 0.01  # the best 0.01
FREEZE = False  # the best False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ex1():
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex1 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE,
                TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    for param in params:
        clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex1 finish %s ************\n\n\n\n' % finish_time)


def ex2():
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex2 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH])
              ]
    for param in params:
        clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex2 finish %s ************\n\n\n\n' % finish_time)


def ex3():
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex3 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE,
                TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    train_paths = ['../data/train.1000.txt', '../data/train.2000.txt', '../data/train.3000.txt',
                   '../data/train.4000.txt', '../data/train.5000.txt']
    for param in params:
        for path in train_paths:
            clf = cf.QuestionClassifier(param[0][0], path, param[0][2], param[0][3], param[0][4], param[0][5])
            clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                      param[1][7], param[1][8])
            print('---------------------------------------------------------------------------------')
        print('#################################################################################')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex3 finish %s ************\n\n\n\n' % finish_time)


def ex4():
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex4 begin %s ************' % begin_time)
    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    params = [([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    for param in params:
        clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        acc, acc_rate = clf.test(test_set, print_detail=True)
        print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
              (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex4 finish %s ************' % finish_time)


print("experiments begin, output transfer to file......")
setup_seed(16)
f = open('../data/output.txt', 'w')
old = sys.stdout
sys.stdout = f

ex1()
print('\n')
ex2()
print('\n')
ex3()
print('\n')
ex4()

sys.stdout = old
f.close()
print("experiments finish, output transfer back!")
