import datetime
import tensorflow
from model import create_model
from data import create_tensors

log_dir = 'harmony/logs/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
model_save_path = 'harmony/trained'


def train():
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    train_xs, train_ys = create_tensors()
    model = create_model()
    history = model.fit(
        train_xs,
        train_ys,
        epochs=150,
        validation_split=0,
        callbacks=[tensorboard_callback]
    )
    model.save(model_save_path)
    print('\n', history, '\n')


if __name__ == '__main__':
    train()
