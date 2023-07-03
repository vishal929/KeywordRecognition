This directory holds logic for the neural network
used for keyword recognition and the main inference loop.

This logic includes:

* Data Processing
* Network Training and Evaluation
* Network Inference to Run on Raspberry Pi 4B
* Bluetooth Serial Listening For Phone Messages
* BLE Peripheral Message Sending To Trigger Microcontroller Switch Flipping

In practice I found that the voice recognition is a little shaky on the raspberry pi,
so I opted to instead listen exclusively on bluetooth serial for commands instead of voice.
To fix this issue, I could implement a sort of button press to initiate recording keywords.

Requirements (See environment.yml for an environment that includes these):

- Python 3.8
- Bleak 0.20.2
- Tensorflow 2.10
- PyBluez 0.30
- PyDub 0.25.1
- Python Sounddevice 0.4.6
- PortAudio 19.6.0

To run the listener and the inference (listen to recording) loop, run main.py

To run just the listener (listen to bluetooth rfcomm serial) run bluetooth_listener.py

Triggers are provided in constants.py with their associated microcontrollers
So, if you want custom triggers, you will have to modify trigger names and microcontrollers in:
1) constants.py
2) MicrocontrollerSystem directory arduino .ino files
