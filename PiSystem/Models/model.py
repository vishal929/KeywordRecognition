import tensorflow as tf
from tensorflow import keras
from PiSystem.constants import LEARN_MAP

# using functional API to build our model
# we are using a convolutional model, which should be channels first (for maximum support across platforms)
# if we have a checkpoint, we load from the checkpoint, otherwise we build from scratch
def build_model(checkPointPath = None):
    # only need a small architecture for keywords
    # B x H x W x C
    # input should already be normalized! (we can use a normalization layer and adapt() for this)
    input = keras.Input(shape=(5,24001,1))

    #print("inputs shape: " + str(input.shape))
    # conv
    x = keras.layers.Conv2D(filters = 32, kernel_size=(3,3),activation = None)(input)
    #print("after conv shape: " + str(x.shape))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # pooling
    x = keras.layers.MaxPool2D(pool_size=(1,2))(x)
    #print("after first pool shape: " + str(x.shape))

    # dropout at low rate
    x = keras.layers.Dropout(rate=0.1)(x)

    x = keras.layers.Conv2D(filters=16, kernel_size=(3, 3))(x)
    #print("after second conv shape: " + str(x.shape))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # pooling
    x = keras.layers.MaxPool2D(pool_size=(1,2))(x)
    #print("after second pool shape: " + str(x.shape))

    # dropout at higher rate before dense
    x = keras.layers.Dropout(rate=0.5)(x)

    # flatten before dense
    x = keras.layers.Flatten()(x)

    # dense to output
    output = keras.layers.Dense(len(LEARN_MAP))(x)

    model = keras.Model(input,output)

    # defining some metrics and a loss function
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(optimizer= "adam", loss = loss)
    return model

# testing the model with input
'''
model = build_model()
test_input = tf.ones(shape=(2,5,24001,1))
out = model.predict(test_input)
print(out.shape)
# 580,854 parameters, this is pretty lightweight hopefully it works nicely for the pi!
print(model.summary())
'''
