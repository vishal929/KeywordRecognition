import tensorflow as tf
from tensorflow import keras
from PiSystem.constants import LEARN_MAP

# using functional API to build our model
# we are using a convolutional model, which should be channels first (for maximum support across platforms)
# if we have a checkpoint, we load from the checkpoint, otherwise we build from scratch
def build_model(checkPointPath = None):
    # I will be using a small resnet like architecture
    # B x H x W x C
    # input should already be normalized! (we can use a normalization layer and adapt() for this)
    input = keras.Input(shape=(None,5,24001,1))

    # 1x1 conv
    x = keras.layers.Conv2D(filters = 16, kernel_size=(3,3),activation = None)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # pooling
    x = keras.layers.MaxPool2D()(x)

    # dropout at low rate
    x = keras.layers.Dropout(rate=0.1)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # pooling
    x = keras.layers.MaxPool2D()(x)

    # dropout at higher rate before dense
    x = keras.layers.Dropout(rate=0.5)(x)

    # dense to output
    output = keras.layers.Dense(len(LEARN_MAP))(x)

    model = keras.Model(input,output)

    # defining some metrics and a loss function
    pass