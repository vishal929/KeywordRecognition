import tensorflow as tf
from PiSystem.constants import LEARN_MAP
import os
from PiSystem.constants import ROOT_DIR


def conv_block(model_input, num_filters, kernel_size, conv_stride, pool_size, dropout_rate=0.2, pool_stride=None,
               padding='same'):
    """
    Definition of a conv block in our model, inspired by the blocks in resnet.
    We use leaky relu, residual connections, batch normalization, and dropout.
    :param model_input: This is the keras input
    :param num_filters: Number of filters to use in our convolutions
    :param kernel_size: Kernel size to use in our convolutions
    :param conv_stride: Stride to use in our convolutions in each dimension
    :param pool_size: spatial size to use for pooling
    :param dropout_rate: Probability of an activation being set to zero in the dropout layer
    :param pool_stride: Stride to use for pooling
    :param padding: padding to apply to pooling and convolution operations
    :return: We return a keras layer output representing the output of this conv block
    """
    # conv
    x = tf.keras.layers.Conv2D(filters=num_filters, padding=padding,
                               kernel_size=kernel_size, strides=conv_stride, activation=None)(model_input)
    # print("after conv shape: " + str(x.shape))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # conv
    x = tf.keras.layers.Conv2D(filters=num_filters, padding=padding,
                               kernel_size=kernel_size, strides=conv_stride, activation=None)(x)
    # print("after conv shape: " + str(x.shape))
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU()(x)

    # skip connection
    res = tf.keras.layers.Conv2D(filters=num_filters, padding=padding, kernel_size=(1, 1), strides=conv_stride,
                                 activation=None)(model_input)
    res = tf.keras.layers.BatchNormalization()(res)
    x = tf.keras.layers.Add()([res, x])

    # pooling across feature dimension
    if pool_size is not None:
        # using conv2d as pooling
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size, padding=padding, strides=pool_stride)(x)
    # print("after pool shape: " + str(x.shape))

    x = tf.keras.layers.LeakyReLU()(x)

    # dropout at low rate
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    return x


def build_model(checkPointPath=None):
    """
    Keras functional API construction of our model
    :param checkPointPath: if we have a checkpoint, we load from the checkpoint, otherwise we build from scratch
    :return: We return a Keras compiled model that we can use for training,evaluation, and inference
    """
    # only need a small architecture for keywords
    # the architecture is inspired by resnet but we are using much fewer conv blocks
    # B x H x W x C
    model_input = tf.keras.Input(shape=(559, 513, 1))
    # normalization layer (we need to call adapt on this before training and saving!)
    x = tf.keras.layers.Normalization(axis=-1)(model_input)  # axis=-1 means we normalize along the channel dimension
    # number of convolutional blocks
    num_blocks = 3
    for i in range(num_blocks):
        # x = conv_block(x,num_filters=16 * ( 2**(i + 1)), kernel_size=(3, 3), conv_stride=(1, 1), pool_size=None)
        # x = conv_block(x, num_filters=16 * (2 ** (i + 1)), kernel_size=(3, 3), conv_stride=(1, 1), pool_size=None)
        x = conv_block(x, num_filters=16 * (2 ** (i + 1)), kernel_size=(3, 3), conv_stride=(1, 1), pool_size=(4, 4))

    # flatten before dense
    x = tf.keras.layers.Flatten()(x)

    # dense to output
    output = tf.keras.layers.Dense(len(LEARN_MAP))(x)

    model = tf.keras.Model(model_input, output)

    # defining some metrics and a loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy()
    ])

    # if we have saved weights, load them
    if checkPointPath:
        model.load_weights(checkPointPath)

    return model


def grab_tflite_model(keras_saved_dir):
    """
    Constructs a tflite model interpreter from our keras trained weights

    :param keras_saved_dir: directory to find the keras model that we have already trained
    :return: We return a tflite interpreter representing the converted tflite model
    """
    tflite_save_dir = os.path.join(ROOT_DIR, 'Models', 'model.tflite')
    if os.path.exists(tflite_save_dir):
        # just load from file
        interpreter = tf.lite.Interpreter(model_path=tflite_save_dir)
        return interpreter

    # saving to file, since it doesnt exist
    model = build_model(keras_saved_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # saving
    with open(tflite_save_dir, 'wb') as f:
        f.write(tflite_model)
    interpreter = tf.lite.Interpreter(model_content=tflite_model)

    return interpreter
