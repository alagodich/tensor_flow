import tensorflow

def create_model():
    print('create_model')
    mnist = tensorflow.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Flatten(input_shape=(28, 28)),
        tensorflow.keras.layers.Dense(128, activation='relu'),
        tensorflow.keras.layers.Dropout(0.2),
        tensorflow.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()

    tensorflow.nn.softmax(predictions).numpy()

    loss_fn = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_fn(y_train[:1], predictions).numpy()

    model.compile(
        optimizer='adam',
        loss=loss_fn,
        metrics=['accuracy']
    )
    print(x_train)
    print(y_train)

    model.fit(x_train, y_train, epochs=1)

    model.evaluate(x_test, y_test, verbose=2)

    probability_model = tensorflow.keras.Sequential([
        model,
        tensorflow.keras.layers.Softmax()
    ])
    probability_model(x_test[:5])

