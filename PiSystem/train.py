# this is the training script
# we will train the model on a different machine than the raspberry pi due to computation concerns
# then we can quantize the model and perform optimizations provided in tensorflow lite before loading on the pi

import tensorflow as tf
from Models.model import build_model
from Data.data_processing import get_dataset, prepare_dataset
from datetime import datetime
import os
from constants import ROOT_DIR

CHECKPOINT_DIR = os.path.join(ROOT_DIR, "Models", "Saved_Checkpoints", "Current_Checkpoint")

'''
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.compat.v1.enable_eager_execution()
'''


class MemoryPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gpu_dict = tf.config.experimental.get_memory_info('GPU:0')
        tf.print('\n GPU memory details [current: {} gb, peak: {} gb]'.format(
            float(gpu_dict['current']) / (1024 ** 3),
            float(gpu_dict['peak']) / (1024 ** 3)))


def train_model(model_checkpoint=None, batch_size=16, learning_rate=0.00005, epochs=300):
    print(tf.config.list_physical_devices('GPU'))

    # seeding the random number generator
    tf.random.set_seed(int(datetime.now().timestamp()))
    model = build_model(model_checkpoint)
    print(model.summary())

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
        train = prepare_dataset(train)
        # removing labels (dont need them for normalization)
        train = train.map(lambda example, label: example)
        #print('actual input shape: ' + str(next(iter(train))))
        # adapting the normalization layer of the model to the dataset
        model.layers[1].adapt(train)
        # model.get_layer("normalization").adapt(list(iter(train)))
        # cleaning up
        del train

    # grabbing our dataset
    train, test = get_dataset()

    train = prepare_dataset(train, set_random_window=True, augment=True, shuffle=True).batch(batch_size=batch_size,
                                                                                             num_parallel_calls=tf.data.AUTOTUNE)

    # print('fit train shape: ' + str(next(iter(train))))
    # need to randomly space clips in the test set, we will repeat the test set a couple of times for this reason
    test = prepare_dataset(test).batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE)

    # fitting (we are using the test data as validation here :) )
    model.fit(train, validation_data=test, epochs=epochs, verbose=2, callbacks=[model_checkpoint_callback,
                                                                                MemoryPrintingCallback()])


train_model()
