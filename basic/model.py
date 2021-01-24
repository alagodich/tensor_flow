import tensorflow

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tensorflow.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension
x_train = x_train[..., tensorflow.newaxis].astype("float32")
x_test = x_test[..., tensorflow.newaxis].astype("float32")

train_ds = tensorflow.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)

test_ds = tensorflow.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        print('Mnist model call')
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = MnistModel()

loss_object = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tensorflow.keras.optimizers.Adam()

train_loss = tensorflow.keras.metrics.Mean(name='train_loss')
train_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tensorflow.keras.metrics.Mean(name='test_loss')
test_accuracy = tensorflow.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tensorflow.function
def train_step(images, labels):
    with tensorflow.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tensorflow.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics each epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}'
        f'Loss {train_loss.result()}'
        f'Accuracy: {train_accuracy.result() * 100}'
        f'Test Loss {test_loss.result()}'
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )


