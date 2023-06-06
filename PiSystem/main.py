# main will hold the inference loop of the pi
'''
    1) detect speech in 3 second windows of "arduino"
    2) if arduino is detected, start to detect for the other keywords like "Stairs" or "Bar"
    3) if within the detection window we detect a keyword, we send a message to the corresponding microcontroller
    4) hopefully microcontroller receives the message properly and flips their designated switch
'''

from Models.model import grab_tflite_model,build_model
import os
from constants import ROOT_DIR,INV_MAP,SAMPLING_RATE
import tensorflow as tf
import numpy as np
import sounddevice as sd
from Data.data_processing import load_audio_file,pad_window

# function which sends a message to the specific microcontroller for this class to flip the switch
def send_message(classname):
    pass

if __name__ == '__main__':

    checkpoint_path = os.path.join(ROOT_DIR,"Models","Saved_Checkpoints","Current_Checkpoint")

    # grab our tflite model interpreter
    interpreter = grab_tflite_model(checkpoint_path)
    base_model = build_model(checkpoint_path)
    print(base_model.summary())
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set constants needed during the inference loop
    arduino_flag = False
    prob_threshold = 0.8

    # setup buffer for holding 3s of audio data
    recording_samples = [np.zeros(SAMPLING_RATE),np.zeros(SAMPLING_RATE),np.zeros(SAMPLING_RATE)]

    # check our sound devices
    print(sd.query_devices())

    while True:
        # inference loop
        # record 1 second of audio and modify the numpy buffers
        new_rec = sd.rec(frames=1*SAMPLING_RATE,samplerate=SAMPLING_RATE,channels=1,dtype='int16',blocking=False)
        recording_samples[0] = recording_samples[1]
        recording_samples[1] = recording_samples[2]

        # waiting on the recording to finish and update buffer
        sd.wait()
        recording_samples[2] = np.squeeze(new_rec)
        #print(recording_samples)
        # need to convert to frequency space to input to model
        tf_in = np.concatenate(recording_samples,axis=None)
        #tf_in = load_audio_file(os.path.join(ROOT_DIR,"Data","Dataset","Test","Gym","Recording.m4a"))
        #tf_in = pad_window(tf_in)
        #tf_in = np.squeeze(sd.rec(frames=3*SAMPLING_RATE,
                                  #samplerate=SAMPLING_RATE,channels=1,dtype='int16',blocking=True))
        tf_in = tf.abs(
            tf.signal.stft(
                tf.convert_to_tensor(tf_in,dtype=tf.float32),
                frame_length = SAMPLING_RATE,
                frame_step = int(SAMPLING_RATE/2),
                fft_length = SAMPLING_RATE,
                pad_end=False
            )
        )
        # need to add channel dimension and batch dimension explicitly for compatibility with model
        tf_in = tf.expand_dims(tf.expand_dims(tf_in,axis=0),axis=-1)
        #print('input shape: ' + str(tf_in.shape))
        #print(tf_in)

        # getting logits for this audio sample of 3 seconds
        #print(input_details)
        interpreter.set_tensor(input_details[0]['index'], tf_in)
        interpreter.invoke()
        logits = tf.squeeze(interpreter.get_tensor(output_details[0]['index']))
        real_logits = base_model.predict(tf_in)
        #print(logits)

        # converting to probablities and seeing if we have some confidence in a class or not
        # we have 5 classes, so a random-confidence is 0.2
        # if we have a confidence in a class of 0.5 or greater, we will consider that as confirmed

        preds = tf.nn.softmax(logits)
        real_preds = tf.nn.softmax(real_logits)
        #print(preds)
        #print(real_preds)
        class_pred = tf.math.argmax(preds).numpy()
        #print(class_pred)
        prob = preds[class_pred]
        detected_class = INV_MAP[class_pred]
        if (not arduino_flag) and prob >= prob_threshold and detected_class == 'arduino':
            # we have triggered the arduino flag, now we have to detect keywords and then send messages
            # this detection window lasts until a valid class is detected
            print("detected keyword: arduino")
            arduino_flag = True
            # resetting the recording array
            recording_samples[0] = np.zeros(SAMPLING_RATE)
            recording_samples[1] = np.zeros(SAMPLING_RATE)
            recording_samples[2] = np.zeros(SAMPLING_RATE)
        elif arduino_flag and prob >= prob_threshold and detected_class != 'arduino':
            # we are in the detection window, and we have detected a keyword
            # lets send a message to the corresponding microcontroller and reset the detection flag
            print("detected keyword during window: " + str(detected_class))
            send_message(detected_class)
            arduino_flag = False
            # resetting the recording array
            recording_samples[0] = np.zeros(SAMPLING_RATE)
            recording_samples[1] = np.zeros(SAMPLING_RATE)
            recording_samples[2] = np.zeros(SAMPLING_RATE)






