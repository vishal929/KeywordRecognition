# main will hold the inference loop of the pi
'''
    1) detect speech in 3 second windows of "arduino"
    2) if arduino is detected, start window to detect for the other keywords like "Stairs" or "Bar"
    3) if within the detection window we detect a keyword, we send a message to the corresponding microcontroller
    4) hopefully microcontroller receives the message properly and flips their designated switch
'''

if __name__ == '__main__':
    print('PyCharm')

