import tensorflow
import numpy
from train import model_save_path
from data import create_tensors
from random import randrange


def format_test_sample_to_chord_string(test_sample):
    quality_list = [
      '+',       '-#5',
      '-7',      '-^7',
      '7',       '7alt',
      '7b13sus', '7b5',
      '7b9sus',  '^',
      '^7#11',   'h',
      'o',       'sus'
    ]
    features = [item[0].astype(int).tolist() for item in test_sample]

    if features[1] == 0:
        shift_label = ''
    else:
        shift_label = 'b' if features[1] == 1 else '#'

    if features[2] == 0:
        quality_label = ''
    else:
        quality_label = quality_list[features[2] - 1]

    return f'Sample chord: {features[0] + 1}{shift_label}{quality_label}'


def predict():
    numpy.set_printoptions(precision=2, suppress=True)
    model = tensorflow.keras.models.load_model(model_save_path)
    data, _ = create_tensors()

    print('\n============================================')
    for i in range(10):
        random_index = randrange(len(data[0]) - 1)
        test_sample = [
            data[0][random_index],
            data[1][random_index],
            data[2][random_index]
        ]
        prediction = model.predict(test_sample)
        print(f'\nFrom {format_test_sample_to_chord_string(test_sample)}')
        print(f'Numeric: {prediction[0][0]}\n')
        print(f'Shift: {prediction[1][0]}\n')
        print(f'Quality: {prediction[2][0]}\n')


if __name__ == '__main__':
    predict()
