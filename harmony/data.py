import pandas
import tensorflow
import numpy


def create_tensors():
    raw_dataset = pandas.read_json('harmony/data/all-jazz-flatten-harmony.json')
    # dataset = raw_dataset.copy()

    train_dataset = raw_dataset.copy()
    # train_dataset = dataset.sample(frac=0.8, random_state=0)
    # test_dataset = dataset.drop(train_dataset.index)

    print(f'Train dataset: \n{train_dataset.describe().transpose()}')

    train_features = train_dataset.copy()

    train_labels = train_features.pop('y')
    train_features = train_features.pop('x')

    # Inputs
    train_features_numeric = [[row[0]] for row in train_features]
    train_features_shift = [[row[1]] for row in train_features]
    train_features_quality = [[row[2]] for row in train_features]

    train_xs = [
        numpy.array(train_features_numeric).astype('float32'),
        numpy.array(train_features_shift).astype('float32'),
        numpy.array(train_features_quality).astype('float32')
    ]

    # Labels
    train_labels_numeric = tensorflow.one_hot([row[0] for row in train_labels], 7)
    train_labels_shift = tensorflow.one_hot([row[1] for row in train_labels], 3)
    train_labels_quality = tensorflow.one_hot([row[2] for row in train_labels], 15)

    train_ys = [
        train_labels_numeric,
        train_labels_shift,
        train_labels_quality
    ]

    return [train_xs, train_ys]


def describe():
    train_xs, train_ys = create_tensors()

    print(train_xs)
    print(train_ys)


if __name__ == '__main__':
    describe()
