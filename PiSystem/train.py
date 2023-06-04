# this is the training script
# we will train the model on a different machine than the raspberry pi due to computation concerns
# then we can quantize the model and perform optimizations provided in tensorflow lite before loading on the pi

import tensorflow as tf
from tensorflow import keras
from Models.model import build_model
from Data.data_processing import get_dataset, augment_train, stft_sound, random_window
from datetime import datetime
import os
from constants import ROOT_DIR

CHECKPOINT_DIR = os.path.join(ROOT_DIR,"Models","Saved_Checkpoints","Current_Checkpoint")

def train(model_checkpoint=None, batch_size=16, learning_rate = 0.001, epochs=300):
    # forcing cpu (for some reason my laptop gpu is failing)
    # Hide GPU from visible devices
    tf.config.set_visible_devices([], 'GPU')
    #print(tf.config.list_physical_devices('GPU'))

    # seeding the random number generator
    keras.utils.set_random_seed(int(datetime.now().timestamp()))
    model = build_model(model_checkpoint)

    model.optimizer.learning_rate.assign(learning_rate)
    print("learning rate: " + str(model.optimizer.learning_rate))

    # setting up callback to save the best model based on validation accuracy
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_DIR,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True)

    # we need to adapt the normalization layer if we did not load weights
    if model_checkpoint is None:
        # getting train data with no augmentation
        train, _ = get_dataset()
        train = (train
                 # need to randomly fit clips less than 3s into a 3s window
                 .map(lambda data, label: tf.py_function(random_window, inp=[data], Tout=[tf.float32]))
                 # convert to frequency domain
                 .map(lambda data: stft_sound(data))
                 # H x W x C format
                 .map(lambda data: tf.expand_dims(tf.squeeze(data), axis=-1))
                 )
        # adapting the normalization layer of the model to the dataset
        model.get_layer("normalization").adapt(train)
        # cleaning up
        del train

    # grabbing our dataset
    train, test = get_dataset()

    train = (train
             .repeat(3)
             # need to randomly fit clips less than 3s into a 3s window
             .map(lambda data, label: (tf.py_function(random_window, inp=[data], Tout=[tf.float32]), label),
                  num_parallel_calls = tf.data.AUTOTUNE)
             .shuffle(buffer_size=100, reshuffle_each_iteration=True)
             # data augmentation
             .map(
                lambda data, label: (tf.py_function(augment_train, inp=[data], Tout=[tf.float32]), label),
                num_parallel_calls = tf.data.AUTOTUNE
             )
             # convert to frequency domain
             .map(lambda data, label: (tf.py_function(stft_sound,inp=[data], Tout=[tf.float32]), label),
                  num_parallel_calls = tf.data.AUTOTUNE
                  )
             # H x W x C format
             .map(lambda data, label: (tf.expand_dims(tf.squeeze(data), axis=-1), label),
                  num_parallel_calls = tf.data.AUTOTUNE
                  )
             .batch(batch_size=batch_size, num_parallel_calls = tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE)
             )


    # need to randomly space clips in the test set, we will repeat the test set a couple of times for this reason
    test = (test
            .repeat(3)
            # need to randomly fit clips less than 3s into a 3s window
            .map(lambda data, label: (tf.py_function(random_window, inp=[data], Tout=[tf.float32]), label),
                 num_parallel_calls = tf.data.AUTOTUNE)
            # convert to frequency domain
            .map(lambda data, label: (tf.py_function(stft_sound,inp=[data],Tout=[tf.float32]), label),
                 num_parallel_calls = tf.data.AUTOTUNE)
            # H x W x C format
            .map(lambda data, label: (tf.expand_dims(tf.squeeze(data), axis=-1), label),
                 num_parallel_calls = tf.data.AUTOTUNE)
            .batch(batch_size=batch_size,num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
            )

    # fitting (we are using the test data as validation here :) )
    model.fit(train, validation_data=test, epochs=epochs, verbose=2, callbacks=[model_checkpoint_callback])


train()
