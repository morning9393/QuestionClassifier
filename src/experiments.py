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
MODEL = 'bilstm'
EMBEDDING_DIM = 200
LSTM_HIDDEN = 100
FC_INPUT = 200
FC_HIDDEN = 64
EPOCHS = 30  # the best 30
LEARNING_RATE = 0.02
FREEZE = False


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def ex1():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex1 begin %s ------------' % begin_time)

    epochs = [1, 2, 3]
    for epoch in epochs:
        bow = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bow.train('bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, FREEZE)

        bilstm = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bilstm.train('bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, FREEZE)

        en_bow = cf.QuestionClassifier(2, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        en_bow.train('bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, FREEZE)

        print('epoch number: %d' % epoch)
        test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        acc, acc_rate = bow.test(test_set)
        print('bow acc: %s' % epoch, str(acc))
        print('bow acc_rate: %s' % epoch, str(acc_rate))

        acc, acc_rate = bilstm.test(test_set)
        print('bilstm acc: %s' % epoch, str(acc))
        print('bilstm acc_rate: %s' % epoch, str(acc_rate))

        acc, acc_rate = en_bow.test(test_set)
        print('en_bow acc: %s' % epoch, str(acc))
        print('en_bow acc_rate: %s' % epoch, str(acc_rate))
    finish_time = time.asctime(time.localtime(time.time()))
    print('ex1 finish %s' % finish_time)


def ex2():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex2 begin %s ------------' % begin_time)

    epochs = [1, 2, 3]
    for epoch in epochs:
        bow_ran = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None)
        bow_ran.train('bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, False)

        bow_pre_fre = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bow_pre_fre.train('bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, True)

        bow_pre_fine = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bow_pre_fine.train('bow', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, False)

        bil_ran = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, None)
        bil_ran.train('bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, False)

        bil_pre_fre = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bil_pre_fre.train('bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, True)

        bil_pre_fine = cf.QuestionClassifier(1, TRAIN_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        bil_pre_fine.train('bilstm', EMBEDDING_DIM, LSTM_HIDDEN, FC_INPUT, FC_HIDDEN, epoch, LEARNING_RATE, False)

        print('epoch number: %d' % epoch)
        test_set = md.QuestionSet(TEST_PATH, VOCABULARY_PATH, LABELS_PATH, STOP_WORDS_PATH, PRE_TRAIN_PATH)
        acc, acc_rate = bow_ran.test(test_set)
        print('bow_ran acc: %s' % str(acc))
        print('bow_ran acc_rate: %s' % str(acc_rate))

        acc, acc_rate = bow_pre_fre.test(test_set)
        print('bow_pre_fre acc: %s' % str(acc))
        print('bow_pre_fre acc_rate: %s' % str(acc_rate))

        acc, acc_rate = bow_pre_fine.test(test_set)
        print('bow_pre_fine acc: %s' % str(acc))
        print('bow_pre_fine acc_rate: %s' % str(acc_rate))

        acc, acc_rate = bil_ran.test(test_set)
        print('bil_ran acc: %s' % str(acc))
        print('bil_ran acc_rate: %s' % str(acc_rate))

        acc, acc_rate = bil_pre_fre.test(test_set)
        print('bil_pre_fre acc: %s' % str(acc))
        print('bil_pre_fre acc_rate: %s' % str(acc_rate))

        acc, acc_rate = bil_pre_fine.test(test_set)
        print('bil_pre_fine acc: %s' % str(acc))
        print('bil_pre_fine acc_rate: %s' % str(acc_rate))
    finish_time = time.asctime(time.localtime(time.time()))
    print('ex2 finish %s' % finish_time)


def ex3():
    begin_time = time.asctime(time.localtime(time.time()))
    print('------------ ex3 begin %s ------------' % begin_time)


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

sys.stdout = old
f.close()
print("experiments finish, output transfer back!")
