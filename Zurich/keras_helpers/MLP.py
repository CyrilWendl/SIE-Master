from __future__ import print_function
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


def get_mlp(n_classes, input_shape, n_filt=70):
    """
    get an MLP model instance
    :param n_classes: number of classes
    :param input_shape: x_train.shape[1:]
    :param n_filt: number of filters per layer
    :return: model
    """
    model_mlp = Sequential()
    model_mlp.add(Dense(n_filt, activation='relu', input_shape=input_shape))
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(n_filt, activation='relu'))
    model_mlp.add(Dropout(0.5))
    model_mlp.add(Dense(n_classes, activation='softmax'))

    model_mlp.summary()

    model_mlp.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
    return model_mlp
