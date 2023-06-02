# holds logic for loading and preprocessing our data
import os
import tensorflow as tf
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
from PiSystem.constants import SAMPLING_RATE,SAMPLE_WIDTH,ROOT_DIR,LEARN_MAP,INV_MAP
from pathlib import Path



# function to return a numpy array of samples from a m4a recording file
def load_audio_file(filename):
    data = AudioSegment.from_file(filename)
    # need to augment the data just for compatibility with different recording devices
    # dealing only with mono input
    data.set_channels(1)
    # our dataset is recorded at 48000 hz and 2 bytes per frame
    data.set_frame_rate(SAMPLING_RATE)
    data.set_sample_width(SAMPLE_WIDTH)
    # returning a numpy array (we have 16 bits but we will use float32 for extra room for augmentation)
    return np.array(data.get_array_of_samples(),dtype=np.float32)

# pipeline to load train and test data (we are not using a validation set since our problem is small in scope)
def get_dataset():
    # firstly getting filenames for train and test data
    train_files = os.path.join(ROOT_DIR,"Data","Dataset","Train","**","*.m4a")
    test_files = os.path.join(ROOT_DIR, "Data", "Dataset", "Test", "**", "*.m4a")

    # getting a dataset of filenames to start
    train = tf.data.Dataset.list_files(train_files)
    test = tf.data.Dataset.list_files(test_files)

    # mapping filenames to their class and transforming filenames to data
    train = train.map(lambda filename: tf.py_function(map_name_to_label_and_data,
                                                      inp=[filename],Tout=[tf.float32,tf.int32]))
    test = test.map(lambda filename: tf.py_function(map_name_to_label_and_data,
                                                    inp=[filename], Tout=[tf.float32,tf.int32]))

    # clip audio to 3 seconds from the beginning (more than enough time to say classnames)
    train = train.map(lambda data,label: (data[:SAMPLING_RATE*3],label))
    test = test.map(lambda data, label: (data[:SAMPLING_RATE * 3], label))

    # training augmentation on audio clips for the training set
    train = train.map(lambda data, label: (tf.py_function(augment_train,inp=[data],Tout=[tf.float32]), label))

    # from experimenting with scaling, scaling the raw sound has an exponential affect on volume
    # I find that a random scale of 0.5 to 3 would be suitable (from quiet but still audible to loud, but not too loud)
    '''
    test_sound = AudioSegment(
        data = (next(iter(train))[0].numpy()[-1,:]).astype(np.int16),
        sample_width= SAMPLE_WIDTH,
        frame_rate=SAMPLING_RATE,
        channels=1
    )
    play(test_sound)
    '''

    # applying short time fourier transform to convert to frequency domain
    train = train.map(lambda data,label: (tf.abs(
                                            tf.signal.stft(data,frame_length=SAMPLING_RATE,
                                                           frame_step=int(SAMPLING_RATE/2),
                                                           fft_length=SAMPLING_RATE,
                                                           pad_end=False)
                                            ),label))
    test = test.map(lambda data, label: (tf.abs(
        tf.signal.stft(data, frame_length=SAMPLING_RATE,
                       frame_step=int(SAMPLING_RATE / 2),
                       fft_length=SAMPLING_RATE,
                       pad_end=False)
    ), label))

    # we end up with data = (5 , 24001), label samples

    return train, test

# logic for mapping filename (absolute) to class label
def map_name_to_label_and_data(filename):
    filename_str = filename.numpy().decode('utf-8')
    print(filename_str)
    path = Path(filename_str)
    classname = path.parent.name

    return load_audio_file(filename_str),LEARN_MAP[classname.strip().lower()]

# augmentation to training data to result in a more robust model
def augment_train(train_example):
    # random scaling for volume (from experimental playback I choose a scale from 0.5 to 3)
    scale = tf.random.uniform(shape=[1],minval=0,maxval=2.5)+0.5
    augmented = tf.multiply(train_example,scale)

    # random noise addition (additive white noise model)
    sumsquare =tf.reduce_sum(tf.pow(train_example,2.0))
    err = sumsquare/ train_example.shape[0]
    stderr = np.sqrt(err)

    noise = tf.random.normal(mean=0.0,stddev=stderr,shape=train_example.shape)
    augmented = tf.add(augmented,noise)
    return augmented



get_dataset()



