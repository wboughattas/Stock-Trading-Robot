import os
import pickle


def import_batches():
    dataset = [unpickle_file('batches.meta'), unpickle_file('data_batch_1'), unpickle_file('data_batch_2'),
               unpickle_file('data_batch_3'), unpickle_file('data_batch_4'), unpickle_file('data_batch_5'),
               unpickle_file('test_batch')]

    return dataset


def unpickle_file(filename):
    data_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'Data', 'CIFAR-10', filename)

    with open(data_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data
