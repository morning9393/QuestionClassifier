import sys
import torch
import random
import numpy as np
import ensemble as en
import model as md


def main(argv):
    """
    Main function, check the input command parameters and choose corresponding handle method.

    :param argv: Command line input parameters.
    """
    if len(argv) < 4:
        print('question_classifier.py [train|test] -config [configuration_file_path]')
        sys.exit(2)
    task = argv[1]
    path = argv[3]
    if argv[2] != '-config' or task not in ('train', 'test'):
        print('question_classifier.py [train|test] -config [configuration_file_path]')
        sys.exit(2)
    config = load_config(path)
    if config['model'] not in ['bow', 'bilstm', 'hybrid-cat', 'hybrid-add', 'cnn']:
        print('model should in [bow|bilstm|hybrid-cat|hybrid-add|cnn]')
    if task == 'train':
        train(config)
    else:
        test(config)


def load_config(path):
    """
    Load config file and transfer to a dictionary.

    :param path: Config file path.
    :return: A dictionary with all configuration values.
    """
    with open(path, 'r') as config_file:
        config = {}
        for line in config_file:
            key, value = line.split('=')
            config[key.strip()] = value.strip()
        config['k'] = int(config['k'])
        config['ensemble_size'] = int(config['ensemble_size'])
        config['embedding_dim'] = int(config['embedding_dim'])
        config['lstm_hidden'] = int(config['lstm_hidden'])
        config['fc_input'] = int(config['fc_input'])
        config['fc_hidden'] = int(config['fc_hidden'])
        config['epoch'] = int(config['epoch'])
        config['learning_rate'] = float(config['learning_rate'])
        config['freeze'] = config['freeze'] == str(True)
        return config


def train(config):
    """
    Train and save model according to config parameters.

    :param config: A dictionary with all configuration.
    """
    classifier = en.QuestionClassifier(config['ensemble_size'], config['train_path'], config['vocabulary_path'],
                                       config['labels_path'], config['stop_words_path'], config['pre_train_path'],
                                       config['k'])
    classifier.train(config['model'], config['embedding_dim'], config['lstm_hidden'], config['fc_input'],
                     config['fc_hidden'], config['epoch'], config['learning_rate'], config['freeze'],
                     config['test_path'])
    classifier.save(config['model_path'])


def test(config):
    """
    Test and save the result according to config parameters.

    :param config: A dictionary with all configuration.
    """
    classifier = en.QuestionClassifier(config['ensemble_size'], config['train_path'], config['vocabulary_path'],
                                       config['labels_path'], config['stop_words_path'], config['pre_train_path'],
                                       config['k'])
    classifier.load(config['model_path'])
    test_set = md.QuestionSet(config['test_path'], config['vocabulary_path'], config['labels_path'],
                              config['stop_words_path'], config['pre_train_path'], config['k'])
    f = open(config['output_path'], 'w')
    old = sys.stdout
    sys.stdout = f
    acc, acc_rate = classifier.test(test_set, is_cnn=(config['model'] == 'cnn'), print_detail=True)
    print('model: %s, ensemble: %d, pre_train: %s, freeze: %s, test_path: %s, acc: %d, acc rate: %f' %
          (config['model'], config['ensemble_size'], config['pre_train_path'], config['freeze'], config['test_path'],
           acc, acc_rate))
    sys.stdout = old
    f.close()


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


if __name__ == "__main__":
    setup_seed(16)
    main(sys.argv)
