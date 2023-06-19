import os

SAMPLING_RATE = 48000 # sampling rate we choose (48000 is the sampling rate of the microphone we are using)
SAMPLE_WIDTH = 2 # number of bytes per sample
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Root Directory


# mapping from classnames to labels used for learning
LEARN_MAP = {
    "arduino":0,
    "bar":1,
    "gymnasium":2,
    "stairs":3,
    "theater":4,
    "silence":5
}

# mapping from learning labels to classnames
INV_MAP = {
    0:"arduino",
    1:"bar",
    2:"gymnasium",
    3:"stairs",
    4:"theater",
    5:"silence"
}

# mapping from switches we want to trigger to device names we encounter in BLE
SWITCH_DEVICE_MAP = {
    "bar": "ItsyBitsy Theater+Bar",
    "theater": "ItsyBitsy Theater+Bar",
    "gymnasium": "ItsyBitsy Gymnasium+Stairs",
    "stairs": "ItsyBitsy Gymnasium+Stairs"
}

# defining some bluetooth uuids for nordic uart service
UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"