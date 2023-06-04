import tensorflow as tf
from tensorflow import keras
from PiSystem.constants import LEARN_MAP


def conv_block(input, num_filters, kernel_size, conv_stride, pool_size, dropout_rate=0.1, pool_stride=None,
               padding='same'):
    # conv
    x = keras.layers.Conv2D(filters=num_filters, padding=padding,
                            kernel_size=kernel_size, strides=conv_stride, activation=None)(input)
    #print("after conv shape: " + str(x.shape))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # conv
    x = keras.layers.Conv2D(filters=num_filters, padding=padding,
                            kernel_size=kernel_size, strides=conv_stride, activation=None)(x)
    #print("after conv shape: " + str(x.shape))
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)

    # skip connection
    res = keras.layers.Conv2D(filters=num_filters, padding=padding, kernel_size=(1,1), strides = conv_stride,
                              activation=None)(input)
    x = keras.layers.Add()([res,x])

    # pooling across feature dimension
    x = keras.layers.MaxPool2D(pool_size=pool_size, padding=padding,strides=pool_stride)(x)
    #print("after pool shape: " + str(x.shape))

    # dropout at low rate
    x = keras.layers.Dropout(rate=dropout_rate)(x)

    return x


# using functional API to build our model
# we are using a convolutional model, which should be channels first (for maximum support across platforms)
# if we have a checkpoint, we load from the checkpoint, otherwise we build from scratch
def build_model(checkPointPath=None):
    # only need a small architecture for keywords
    # the architecture is inspired by resnet but we are using much fewer conv blocks
    # B x H x W x C
    input = keras.Input(shape=(5, 24001, 1))
    # normalization layer (we need to call adapt on this before training and saving!)
    x = keras.layers.Normalization(axis=-1)(input) # axis=-1 means we normalize along the channel dimension
    # number of convolutional blocks
    num_blocks = 10
    for i in range(num_blocks):
        if i == num_blocks-1:
            x = conv_block(x,num_filters=32 * (i + 1), kernel_size=(3, 3), conv_stride=(1, 1), pool_size=(1, 2),
                           dropout_rate=0.5)
        else:
            x = conv_block(x,num_filters=32 * (i + 1), kernel_size=(3, 3), conv_stride=(1, 1), pool_size=(1, 2))

    # flatten before dense
    x = keras.layers.Flatten()(x)

    # dense to output
    output = keras.layers.Dense(len(LEARN_MAP))(x)

    model = keras.Model(input, output)

    # defining some metrics and a loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss, metrics=[
        keras.metrics.SparseCategoricalAccuracy()
    ])

    # if we have saved weights, load them
    if checkPointPath:
        model.load_weights(checkPointPath)

    return model

'''
# testing the model with input
model = build_model()
test_input = tf.ones(shape=(2, 5, 24001, 1))
out = model.predict(test_input)
print(out.shape)
# resnet 18 can do 3 fps on the raspberry pi (this is actually more than we need, but I want to make my own arch)
# we want a single computation every second, which means we just need a model that can do 1fps)
# however I also want the raspberry pi to have enough compute left in the time window to send messages to the microcontroller
# anything below resnet 18 parameters (~11M) is fair game
print(model.summary())
'''
