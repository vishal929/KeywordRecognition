import os

SAMPLING_RATE = 16000 # sampling rate we choose
SAMPLE_WIDTH = 2 # number of bytes per sample
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Root Directory


# mapping from classnames to labels used for learning
LEARN_MAP = {
    "arduino":0,
    "bar":1,
    "gym":2,
    "stairs":3,
    "theater":4,
    "other":5
}

# mapping from learning labels to classnames
INV_MAP = {
    0:"arduino",
    1:"bar",
    2:"gym",
    3:"stairs",
    4:"theater",
    5:"other"
}