# main will hold the inference loop of the pi
'''
    1) detect speech in 3 second windows of "arduino"
    2) if arduino is detected, start to detect for the other keywords like "Stairs" or "Bar"
    3) if within the detection window we detect a keyword, we send a message to the corresponding microcontroller
    4) hopefully microcontroller receives the message properly and flips their designated switch
'''

from Models.model import grab_tflite_model
import os
from constants import ROOT_DIR,INV_MAP,SAMPLING_RATE
import tensorflow as tf
import numpy as np
import sounddevice as sd

# function which sends a message to the specific microcontroller for this class to flip the switch
def send_message(classname):
    pass

if __name__ == '__main__':

    checkpoint_path = os.path.join(ROOT_DIR,"Models","Saved_Checkpoints","Final_Model")

    # grab our tflite model interpreter
    interpreter = grab_tflite_model(checkpoint_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set flags needed during the inference loop
    arduino_flag = False

    # setup buffer for holding 3s of audio data
    recording_samples = np.zeros((3,SAMPLING_RATE))

    # check our sound devices
    print(sd.query_devices())

    while True:
        # inference loop
        # record 1 second of audio and modify the numpy buffers
        new_rec = sd.rec(frames=1*SAMPLING_RATE,samplerate=SAMPLING_RATE,channels=1,dtype=np.float32)
        recording_samples[0] = recording_samples[1]
        recording_samples[1] = recording_samples[2]

        # waiting on the recording to finish and update buffer
        sd.wait()
        recording_samples[2] = new_rec

        # getting logits for this audio sample of 3 seconds
        interpreter.set_tensor(input_details[0]['index'], np.concatenate(recording_samples))
        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]['index'])
        print(logits)

        # converting to probablities and seeing if we have some confidence in a class or not
        # we have 5 classes, so a random-confidence is 0.2
        # if we have a confidence in a class of 0.5 or greater, we will consider that as confirmed

        preds = tf.nn.softmax(logits)
        class_pred = tf.math.argmax(preds)
        prob = preds[class_pred]
        detected_class = INV_MAP[class_pred]
        if (not arduino_flag) and prob >= 0.5 and detected_class == 'arduino':
            # we have triggered the arduino flag, now we have to detect keywords and then send messages
            # this detection window lasts until a valid class is detected
            arduino_flag = True
        elif arduino_flag and prob >=0.5 and detected_class != 'arduino':
            # we are in the detection window, and we have detected a keyword
            # lets send a message to the corresponding microcontroller and reset the detection flag
            send_message(detected_class)
            arduino_flag = False






