import sys
import torch
import random
import time
import numpy as np
from src import question_classifier as cf
from src import model as md

TRAIN_PATH = '../data/train.5000.txt'
DEV_PATH = '../data/dev.txt'
TEST_PATH = '../data/test.txt'
VOCABULARY_PATH = '../data/vocabulary.txt'
LABELS_PATH = '../data/labels.txt'
STOP_WORDS_PATH = '../data/stop_words.txt'
PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
ENSEMBLE_SIZE = 20  # the best 20
MODEL = 'hybrid-cat'  # the best hybrid-cat
EMBEDDING_DIM = 200  # the best 200
LSTM_HIDDEN = 100  # the best 100
FC_INPUT = 200  # the best 200 / 400 for hybrid-cat / 784 for cnn
FC_HIDDEN = 64  # the best 64
EPOCHS = 30  # the best 30
LEARNING_RATE = 0.02  # the best 0.02
FREEZE = False  # the best False
IS_CNN = False
PRINT_DETAIL = False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ex1():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex1 begin %s ------------' % begin_time)

    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [True, PRINT_DETAIL]),
              ([20, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL])
              ]

    epochs = [1, 5, 10, 15, 20, 25, 30]
    for epoch in epochs:
        print("number of epoch: %d" % epoch)
        for param in params:
            clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
            clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], epoch, param[1][6], param[1][7])
            acc, acc_rate = clf.test(test_set, params[2][0], params[2][1])
            print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
                  (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))

    finish_time = time.asctime(time.localtime(time.time()))
    print('ex1 finish %s' % finish_time)


def ex2():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex2 begin %s ------------' % begin_time)

    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False],
               [IS_CNN, PRINT_DETAIL])
              ]

    epochs = [1, 5, 10, 15, 20, 25, 30]
    for epoch in epochs:
        print("number of epoch: %d" % epoch)
        for param in params:
            clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
            clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], epoch, param[1][6], param[1][7])
            acc, acc_rate = clf.test(test_set, params[2][0], params[2][1])
            print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
                  (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))

    finish_time = time.asctime(time.localtime(time.time()))
    print('ex2 finish %s' % finish_time)


def ex3():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex3 begin %s ------------' % begin_time)

    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [True, PRINT_DETAIL]),
              ([20, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, PRINT_DETAIL])
              ]

    train_paths = ['../data/train.1000.txt', '../data/train.2000.txt', '../data/train.3000.txt',
                   '../data/train.4000.txt', '../data/train.5000.txt', ]
    for path in train_paths:
        print("train set: %s" % path)
        for param in params:
            clf = cf.QuestionClassifier(param[0][0], path, param[0][2], param[0][3], param[0][4], param[0][5])
            clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                      param[1][7])
            acc, acc_rate = clf.test(test_set, params[2][0], params[2][1])
            print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
                  (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))

    finish_time = time.asctime(time.localtime(time.time()))
    print('ex3 finish %s' % finish_time)


def ex4():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex4 begin %s ------------' % begin_time)

    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
    params = [([20, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE],
               [IS_CNN, True])
              ]

    for param in params:
        clf = cf.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7])
        acc, acc_rate = clf.test(test_set, params[2][0], params[2][1])
        print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
              (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))

    finish_time = time.asctime(time.localtime(time.time()))
    print('ex4 finish %s' % finish_time)


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
