import tensorflow as tf
from tensorflow import keras

# using functional API to build our model
# we are using a convolutional model, which should be channels first (for maximum support across platforms)
# if we have a checkpoint, we load from the checkpoint, otherwise we build from scratch
def build_model(checkPointPath = None):
    pass