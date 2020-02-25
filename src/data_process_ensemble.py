def parse_labels(path):
    label_layer = {}
    with open(path, 'r') as label_file:
        for line in label_file:
            main_label, sub_label = line.strip().split(":")
            if main_label in label_layer.keys():
                label_layer[main_label].append(sub_label)
            else:
                label_layer[main_label] = [sub_label]
    return label_layer


def write_parsed_labels(label_layer):
    with open('../data/label_main.txt', 'w') as label_main_file:
        label_main_file.write('\n'.join(label_layer.keys()))
    for main_label in label_layer.keys():
        sub_label_path = '../data/label_' + main_label + '.txt'
        with open(sub_label_path, 'w') as label_sub_file:
            label_sub_file.write('\n'.join(label_layer[main_label]))


def parse_data(path):
    data_layer = {}
    with open(path, 'r') as data_file:
        for line in data_file:
            label, question = line.strip().split(' ', 1)
            main_label, sub_label = label.split(':')
            if main_label in data_layer.keys():
                if sub_label in data_layer[main_label].keys():
                    data_layer[main_label][sub_label].append(question)
                else:
                    data_layer[main_label][sub_label] = [question]
            else:
                data_layer[main_label] = {}
                data_layer[main_label][sub_label] = [question]
    return data_layer


def write_parsed_data(data_layer, prefix):
    main_data = []
    with open('../data/' + prefix + '_main.txt', 'w') as data_main_file:
        for main_label in data_layer.keys():
            for sub_label in data_layer[main_label].keys():
                for question in data_layer[main_label][sub_label]:
                    main_data.append(main_label + ' ' + question)
        data_main_file.write('\n'.join(main_data))
    for main_label in data_layer.keys():
        sub_data_path = '../data/' + prefix + '_' + main_label + '.txt'
        sub_data = []
        with open(sub_data_path, 'w') as data_sub_file:
            for sub_label in data_layer[main_label].keys():
                for question in data_layer[main_label][sub_label]:
                    sub_data.append(sub_label + ' ' + question)
            data_sub_file.write('\n'.join(sub_data))


write_parsed_labels(parse_labels('../data/labels.txt'))
write_parsed_data(parse_data('../data/train.txt'), 'train')
write_parsed_data(parse_data('../data/dev.txt'), 'dev')

