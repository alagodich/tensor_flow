import tensorflow


def create_model():
    numeric_input = tensorflow.keras.Input(name='numeric_input', shape=(1,))
    shift_input = tensorflow.keras.Input(name='shift_input', shape=(1,))
    quality_input = tensorflow.keras.Input(name='quality_input', shape=(1,))

    inputs = [numeric_input, shift_input, quality_input]

    numeric_embedding = tensorflow.keras.layers.Embedding(
        name='numeric_embedding',
        input_dim=7,
        output_dim=2,
        input_length=1
    )(numeric_input)
    shift_embedding = tensorflow.keras.layers.Embedding(
        name='shift_embedding',
        input_dim=3,
        output_dim=2,
        input_length=1
    )(shift_input)
    quality_embedding = tensorflow.keras.layers.Embedding(
        name='quality_embedding',
        input_dim=15,
        output_dim=4,
        input_length=1
    )(quality_input)

    shared_input_layer = tensorflow.keras.layers.Concatenate()([
        tensorflow.keras.layers.Flatten()(numeric_embedding),
        tensorflow.keras.layers.Flatten()(shift_embedding),
        tensorflow.keras.layers.Flatten()(quality_embedding)
    ])

    shared_hidden_layer = tensorflow.keras.layers.Dense(units=200, activation='relu')(shared_input_layer)

    outputs = [
        tensorflow.keras.layers.Dense(name='numeric_output', units=7, activation='softmax')(shared_hidden_layer),
        tensorflow.keras.layers.Dense(name='shift_output', units=3, activation='softmax')(shared_hidden_layer),
        tensorflow.keras.layers.Dense(name='quality_output', units=15, activation='softmax')(shared_hidden_layer)
    ]

    model = tensorflow.keras.Model(inputs=inputs, outputs=outputs, name='harmony_model')
    model.compile(
        optimizer=tensorflow.keras.optimizers.Adam(),
        loss=tensorflow.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


def describe():
    model = create_model()

    model.summary()
    print('Model inputs:')
    [print(i.shape, i.dtype) for i in model.inputs]
    print('\nModel outputs:')
    [print(i.shape, i.dtype) for i in model.outputs]

    tensorflow.keras.utils.plot_model(model, 'harmony/data/harmony_model.png', show_shapes=True)


if __name__ == '__main__':
    describe()
