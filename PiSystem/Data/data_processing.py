# holds logic for loading and preprocessing our data
import os
import tensorflow as tf
from pydub import AudioSegment
import numpy as np
from PiSystem.constants import SAMPLING_RATE, SAMPLE_WIDTH, ROOT_DIR, LEARN_MAP, INV_MAP
from pathlib import Path


# function to return a numpy array of samples from a m4a recording file
def load_audio_file(filename):
    data = AudioSegment.from_file(filename)
    # need to augment the data just for compatibility with different recording devices
    # dealing only with mono input
    data.set_channels(1)
    # downsampling if needed based on constants
    data.set_frame_rate(SAMPLING_RATE)
    data.set_sample_width(SAMPLE_WIDTH)
    # returning a numpy array (we have 16 bits but we will use float32 for extra room for augmentation)
    out = np.array(data.get_array_of_samples(), dtype=np.float32)
    # clipping the audio to 3s if larger, will do padding after
    if out.shape[0] > 3 * SAMPLING_RATE:
        out = out[:3 * SAMPLING_RATE]

    return tf.convert_to_tensor(out,dtype=tf.float32)


# pipeline to load train and test data (we are not using a validation set since our problem is small in scope)
# if augment is true, we randomly place the audio segment for the training set into a 3s window
# if augment is false, we pad the end of the segment until the window is 3s long
def get_dataset():
    # firstly getting filenames for train and test data
    train_files = os.path.join(ROOT_DIR, "Data", "Dataset", "Train", "**", "*.m4a")
    test_files = os.path.join(ROOT_DIR, "Data", "Dataset", "Test", "**", "*.m4a")

    # getting a dataset of filenames to start
    train = tf.data.Dataset.list_files(train_files)
    test = tf.data.Dataset.list_files(test_files)

    # mapping filenames to their class and transforming filenames to data
    train = train.map(lambda filename: tf.py_function(map_name_to_label_and_data,
                                                      inp=[filename], Tout=[tf.float32, tf.int32]),
                      num_parallel_calls=tf.data.AUTOTUNE)
    test = test.map(lambda filename: tf.py_function(map_name_to_label_and_data,
                                                    inp=[filename], Tout=[tf.float32, tf.int32]),
                    num_parallel_calls=tf.data.AUTOTUNE)

    # audio segments at this point are clipped to 3s, not padded!
    return train, test

# prepare a tf.dataset for training and testing (we should call batch() on the output of this)
def prepare_dataset(dataset,set_random_window=False,augment=False,shuffle=False):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=300, reshuffle_each_iteration=True)
    if augment:
        # need to repeat samples if we are doing augmentation
        dataset = dataset.repeat(3)
    if set_random_window:
        # need to apply the random windowing augmentation
        dataset = dataset.map(lambda example,label: (tf.py_function(random_window,inp=[example],Tout=[tf.float32]),label),
                          num_parallel_calls = tf.data.AUTOTUNE)
    else:
        # we just pad the end with zeros
        dataset = dataset.map(lambda example,label: (tf.py_function(pad_window,inp=[example],Tout=[tf.float32]),label),
                          num_parallel_calls = tf.data.AUTOTUNE)

    # data augmentation after padding
    if augment:
       dataset = dataset.map(
                lambda data, label: (augment_train(data), label),
                num_parallel_calls = tf.data.AUTOTUNE
             )
    # convert to frequency domain
    dataset = dataset.map(lambda data, label: (stft_sound(data), label),
                  num_parallel_calls = tf.data.AUTOTUNE
            )
    # H x W x C format
    dataset = dataset.map(lambda data, label: (tf.expand_dims(tf.squeeze(data), axis=-1), label),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# logic for converting to the frequency domain
def stft_sound(data):
    # I want to use the cpu for stft (for some reason I get oom with gpu)
    with tf.device('/cpu:0'):
        # print("stft input: " + str(data.shape))
        stft = tf.signal.stft(data, frame_length=SAMPLING_RATE,
                              frame_step=int(SAMPLING_RATE / 2),
                              fft_length=SAMPLING_RATE,
                              pad_end=False)
        #print('after stft shape: ' + str(stft.shape))
        mag = tf.squeeze(tf.abs(stft))
        #print('after abs shape: ' + str(mag.shape))
        return mag


# logic for mapping filename (absolute) to class label
def map_name_to_label_and_data(filename):
    filename_str = filename.numpy().decode('utf-8')
    path = Path(filename_str)
    classname = path.parent.name

    return load_audio_file(filename_str), LEARN_MAP[classname.strip().lower()]


# just padding the end of a sample to fit 3 seconds in a window
def pad_window(example):
    with tf.device('/cpu:0'):
        # print("pad window: " + str(example.shape))
        if (example.shape[0] < 3 * SAMPLING_RATE):
            # need to pad end of the sound clip to the desired length
            paddings = tf.constant([[0, 3 * SAMPLING_RATE - example.shape[0]]])
            # paddings[1] = 3* SAMPLING_RATE - example.shape[0]
            return tf.pad(tf.squeeze(example), paddings)
        else:
            return tf.squeeze(example[:3 * SAMPLING_RATE])


# need to fit samples which are less than 3 seconds long into a window of 3 seconds
def random_window(example):
    with tf.device('/cpu:0'):
        # if the audio clip is less than 3 seconds (which most are), we can randomly place this in a window of 3 seconds
        if (example.shape[0] < 3 * SAMPLING_RATE):
            window = np.zeros(shape=(3 * SAMPLING_RATE), dtype=np.float32)

            # choose a random start index from 0 to len(window)-len(train_example)
            rand_index = tf.random.uniform(shape=[1], minval=0, maxval=3 * SAMPLING_RATE - example.shape[0],
                                           dtype=tf.int32).numpy()[0]
            window[rand_index:rand_index + example.shape[0]] = example.numpy()
            augmented = tf.convert_to_tensor(window, dtype=tf.float32)

            return augmented
        else:
            return example


'''
    data augmentation on train data to result in a more robust model
    1) random scaling to adjust gain
    2) random additive noise based on sum squared of given samples
'''
def augment_train(train_example):
    # print('augment train example shape: ' + str(train_example))
    # random scaling for volume (from experimental playback I choose a scale from 0.5 to 3)
    scale = tf.random.uniform(shape=[1], minval=0, maxval=2.5) + 0.5
    augmented = tf.multiply(train_example, scale)

    # random noise addition (additive white noise model)
    num_elements = 3 * SAMPLING_RATE
    sumsquare = tf.reduce_sum(tf.pow(augmented, 2.0))
    err = sumsquare / num_elements
    stderr = tf.sqrt(err)

    # adding in the random noise
    noise = tf.random.normal(mean=0.0, stddev=stderr, shape=[num_elements])
    augmented = tf.add(augmented, noise)
    return augmented

