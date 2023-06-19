# main will hold the inference loop of the pi
'''
    1) detect speech in 3 second windows of "arduino"
    2) if arduino is detected, start to detect for the other keywords like "Stairs" or "Bar"
    3) if within the detection window we detect a keyword, we send a message to the corresponding microcontroller
    4) hopefully microcontroller receives the message properly and flips their designated switch
'''
import asyncio
import queue
from threading import Thread
from time import time

from Models.model import grab_tflite_model
import os
from constants import ROOT_DIR,INV_MAP,SAMPLING_RATE
import tensorflow as tf
import numpy as np
import sounddevice as sd
from Messaging.message import send_message
from Listener.listen import ListenerService, uuid, ListenThread
from multiprocessing import Lock,Process



async def main():
    checkpoint_path = os.path.join(ROOT_DIR,"Models","Saved_Checkpoints","Current_Checkpoint")

    # initialize our BLE connections to our nrf58240 boards
    #connection_manager = BLEConnectionManager()

    # grab our tflite model interpreter
    interpreter = grab_tflite_model(checkpoint_path)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set constants needed during the inference loop
    arduino_flag = False
    arduino_win_count = 0
    prob_threshold = 0.7

    # setup buffer for holding 3s of audio data
    recording_samples = np.zeros(3*SAMPLING_RATE)
    recording_queue = queue.Queue(maxsize=12)

    # check our sound devices
    print(sd.query_devices())

    # setting up callback function for audio recording
    def record_callback(indata, frames,time, status):
        recording_queue.put(np.copy(indata[:,0]))

    # setup continuous recording callback
    stream = sd.InputStream(samplerate=SAMPLING_RATE,blocksize=1*SAMPLING_RATE,channels=1,callback=record_callback,dtype='int16')

    # we want to setup a listener to listen to phone messages via ble to trigger switches also!
    # this can be done via nrf connect or adafruit connect apps through the uart writers
    # we need a lock for ble
    ble_lock = Lock()
    listen_queue = queue.Queue()
    listen_queue.put(ble_lock,block=True)
    listen_thread = ListenThread(listen_queue)
    listen_thread.start()

    # counting the time for windows
    s = time()
    with stream:
        while True:
            recording_sample = None
            try:
                # queue processing
                recording_sample = recording_queue.get_nowait()
                # if the queue is full, just dump it
                if recording_queue.full():
                    with recording_queue.mutex:
                        # just replace the entire queue
                        recording_queue = queue.Queue(maxsize=12)
            except queue.Empty as e:
                # queue has no new elements we continue
                continue
            # update our buffer
            # we first roll the old values, then update the end of the buffer with the new values
            recording_samples = np.roll(recording_samples,-recording_sample.shape[0])
            # updating the most recent part of the buffer with the new elements
            recording_samples[-recording_sample.shape[0]:] = recording_sample

            # need to convert to frequency space to input to model
            tf_in = recording_samples

            tf_in = tf.abs(
                tf.signal.stft(
                    tf.convert_to_tensor(tf_in, dtype=tf.float32),
                    frame_length=1024,
                    frame_step=256,
                    pad_end=False
                )
            )
            # need to add channel dimension and batch dimension explicitly for compatibility with model
            tf_in = tf.expand_dims(tf.expand_dims(tf_in, axis=0), axis=-1)

            # getting logits for this audio sample of 3 seconds
            interpreter.set_tensor(input_details[0]['index'], tf_in)
            interpreter.invoke()
            logits = tf.squeeze(interpreter.get_tensor(output_details[0]['index']))

            # converting to probablities and seeing if we have some confidence in a class or not
            # we have 5 classes, so a random-confidence is 0.2
            # if we have a confidence in a class of 0.7 or greater, we will consider that as confirmed

            preds = tf.nn.softmax(logits)
            class_pred = tf.math.argmax(preds).numpy()
            prob = preds[class_pred]
            detected_class = INV_MAP[class_pred]
            if prob >= prob_threshold:
                if (not arduino_flag) and detected_class == 'arduino':
                    # we have triggered the arduino flag, now we have to detect keywords and then send messages
                    # this detection window lasts for 5s or until a valid class is detected
                    print('arduino window initiated with probability: ' + str(prob))
                    arduino_flag = True
                    # resetting the time and our buffer
                    recording_samples = np.zeros(3*SAMPLING_RATE)
                    s = time()
                    continue
                elif arduino_flag and detected_class != 'arduino' and detected_class != 'silence':
                    # we are in the detection window, and we have detected a keyword that is not silence
                    # lets send a message to the corresponding microcontroller and reset the detection flag
                    print('sending a message to class: ' + str(detected_class) + ' with probability: ' + str(prob))
                    Thread(target=send_message, args=(detected_class,)).start()
                    #ble_lock.acquire()
                    #connection_manager.send_message(detected_class)
                    #ble_lock.release()
                    arduino_flag = False
                    arduino_win_count = 0
                    print('arduino window stopped due to class given')
                    # resetting the time and our buffer
                    recording_samples = np.zeros(3 * SAMPLING_RATE)
                    s = time()
                    continue

            # counting time
            if arduino_flag:
                new_time = time()
                # updating elapsed time
                arduino_win_count += new_time-s
                s = new_time
                if arduino_win_count >= 6:
                    # want to stop the window after 6 seconds has passed without a valid command
                    arduino_flag = False
                    arduino_win_count = 0
                    print('arduino window stopped due to time window')

if __name__ == '__main__':
    asyncio.run(main())