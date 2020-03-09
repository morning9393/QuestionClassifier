import sys
import torch
import random
import time
import numpy as np
import ensemble as en
import model as md

TRAIN_PATH = '../data/train.5000.txt'
DEV_PATH = '../data/dev.txt'
TEST_PATH = '../data/test.txt'
VOCABULARY_PATH = '../data/vocabulary.txt'
LABELS_PATH = '../data/labels.txt'
STOP_WORDS_PATH = '../data/stop_words.txt'
PRE_TRAIN_PATH = '../data/glove.200d.small.txt'
K = 3
ENSEMBLE_SIZE = 20
MODEL = 'hybrid-cat'
EMBEDDING_DIM = 200  # 200d or 300d
LSTM_HIDDEN = 100  # For 300d: 150. For 200d: 100
FC_INPUT = 200  # For 300d: 300 for others / 600 for hybrid-cat / 1184 for cnn. For 200d: 200, 400, 784
FC_HIDDEN = 64
EPOCHS = 30
LEARNING_RATE = 0.01
FREEZE = False


def setup_seed(seed):
    """
    Fix random process.

    :param seed: Seed of random value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ex1():
    """
    First experiment, compare performance of different models with same parameters and epochs.
    """
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex1 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE,
                TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    for param in params:
        clf = en.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],
                                    param[0][6])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex1 finish %s ************\n\n\n\n' % finish_time)


def ex2():
    """
    Second experiment, compare the influence of different initial parameters based on bow and bilstm,
    including [random initialized/pre-trained&freeze/pre-trained&fine-tune].
    """
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex2 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None, K],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, True, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, False, TEST_PATH])
              ]
    for param in params:
        clf = en.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],
                                    param[0][6])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex2 finish %s ************\n\n\n\n' % finish_time)


def ex3():
    """
    Third experiment, compare performance of each model under train set with different size.
    """
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex3 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-add', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE,
                TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['cnn', EMBEDDING_DIM, LSTM_HIDDEN, 784, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    train_paths = ['../data/train.1000.txt', '../data/train.2000.txt', '../data/train.3000.txt',
                   '../data/train.4000.txt', '../data/train.5000.txt']
    for param in params:
        for path in train_paths:
            clf = en.QuestionClassifier(param[0][0], path, param[0][2], param[0][3], param[0][4], param[0][5],
                                        param[0][6])
            clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                      param[1][7], param[1][8])
            print('---------------------------------------------------------------------------------')
        print('#################################################################################')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex3 finish %s ************\n\n\n\n' % finish_time)


def ex4():
    """
    Forth experiment, choose the most accurate model, check the classifying result for every test sample.
    """
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex4 begin %s ************' % begin_time)
    test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K)
    params = [([ENSEMBLE_SIZE, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, K],
               ['hybrid-cat', EMBEDDING_DIM, LSTM_HIDDEN, 400, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    for param in params:
        clf = en.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],
                                    param[0][6])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        acc, acc_rate = clf.test(test_set, print_detail=True)
        print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, acc: %d, acc rate: %f' %
              (param[1][0], param[0][0], param[0][5], param[1][7], acc, acc_rate))
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex4 finish %s ************\n\n\n\n' % finish_time)


def ex5():
    """
    Fifth experiment, compare the influence of different data preprocess based on bow model.
    """
    begin_time = time.asctime(time.localtime(time.time()))
    print('************ ex5 begin %s ************' % begin_time)
    params = [([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, 1],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, 3],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH, 5],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH]),
              ([1, TRAIN_PATH, '../data/vocabulary_empty_stop_words.txt', LABELS_PATH, '../data/stop_words_empty.txt',
                PRE_TRAIN_PATH, K],
               ['bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, EPOCHS, LEARNING_RATE, FREEZE, TEST_PATH])
              ]
    for param in params:
        clf = en.QuestionClassifier(param[0][0], param[0][1], param[0][2], param[0][3], param[0][4], param[0][5],
                                    param[0][6])
        clf.train(param[1][0], param[1][1], param[1][2], param[1][3], param[1][4], param[1][5], param[1][6],
                  param[1][7], param[1][8])
        print('---------------------------------------------------------------------------------')
    finish_time = time.asctime(time.localtime(time.time()))
    print('************ ex5 finish %s ************' % finish_time)


if __name__ == "__main__":
    print("experiments begin, output transfer to file [../data/experiments_output.txt]......")
    setup_seed(16)
    f = open('../data/experiments_output.txt', 'w')
    old = sys.stdout
    sys.stdout = f

    ex1()
    ex2()
    ex3()
    ex4()
    ex5()

    sys.stdout = old
    f.close()
    print("experiments finish, output transfer back!")
