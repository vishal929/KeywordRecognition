# logic for sending messages to specific microcontrollers and connecting to them
from time import sleep

from adafruit_ble import BLERadio
from PiSystem.constants import SWITCH_DEVICE_MAP
from adafruit_ble.services.nordic import UARTService
import threading
import re


class BLEConnectionManager:
    """
        Constructor for our BLE Connection Manager
        Attempts to connect and pair to our Adafruit nrf52840's on initialization
        :param timeout: Time in seconds until we should consider connection attempts failed
    """
    def __init__(self, timeout=10):
        # we should have a radio to manage connections
        self.radio = BLERadio()
        # we should set a timeout of seconds until the connection is considered failed
        self.timeout = timeout
        # we should have a lock for synchronized access
        self.lock = threading.Lock()

    """
        This function serves to discover a device on BLE by name so we can connect and send a message
        We use the switch name to device name mapping defined in constants.py to manage this
        :param device_name: this is a string of the keyword that should be present in the device name
                            I.E itsybitsy theater+bar
        :param timeout: this is the number of seconds to search before timing out
    """
    def discover_device(self,device_name):
        found = set()
        for entry in self.radio.start_scan(timeout=self.timeout):
            addr = entry.address
            if addr not in found:
                if entry.complete_name and device_name in entry.complete_name:
                    print('complete name: ' + str(entry.complete_name))
                    self.radio.stop_scan()
                    return addr
                found.add(addr)
        return None

    """
        This function serves to send a message to a device to trigger a switch based on the class
        :param class_name: this is the name of the switch to trigger, i.e "bar","theater", etc. defined in constants.py
    """
    def send_message(self,class_name):
        self.lock.acquire()
        # firstly filtering the class_name to be lowercase alphabetical characters only
        class_name = class_name.strip()
        filtered = ""
        for c in class_name:
            if c.isalpha():
                filtered += c.lower()
        class_name = filtered
        print('received processed key: ' + str(class_name))
        # need to check if this is a valid class
        if class_name not in SWITCH_DEVICE_MAP:
            print('invalid key -> will abort message sending...')
            self.lock.release()
            return
        device = SWITCH_DEVICE_MAP[class_name]
        try:
            addr = self.discover_device(device)
            if addr is None:
                print(' could not discover device: ' + str(device) +' -> will abort message sending...')
                self.lock.release()
                return
            conn = self.radio.connect(addr, timeout=self.timeout)
            # sending the actual message
            conn[UARTService].write(class_name.lower().encode('ascii'))
            # disconnecting from the controller
            conn.disconnect()
        finally:
            self.lock.release()

'''
manager = BLEConnectionManager()
while True:
    manager.send_message("theater")
    sleep(4)
    manager.send_message("bar")
    sleep(4)
    manager.send_message(' b;;;; a ;;;; r;;; ')
'''
