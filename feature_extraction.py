import pickle
import numpy as np
import tensorflow as tf
# TODO: import Keras layers you need here
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_integer('num_epochs', 30, "number of epochs")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    num_classes = len(set(np.squeeze(y_train))) # == len(np.unique(y_train))

    print('Train set shape:', X_train.shape, y_train.shape)
    print('Valid set shape:', X_val.shape, y_val.shape)
    print('Num classes:', num_classes)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # TODO: train your model here
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, nb_epoch=FLAGS.num_epochs, batch_size=FLAGS.batch_size, validation_data=(X_val, y_val), shuffle=True)

    #score = model.evaluate(X_val, y_val, batch_size=FLAGS.batch_size)
    K.clear_session()

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
