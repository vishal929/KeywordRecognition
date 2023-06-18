# logic for sending messages to specific microcontrollers and connecting to them
from time import sleep

from adafruit_ble import BLERadio
from PiSystem.constants import SWITCH_DEVICE_MAP
from adafruit_ble.services.nordic import UARTService


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
        device = SWITCH_DEVICE_MAP[class_name]
        addr = self.discover_device(device)
        if addr is None:
            return
        conn = self.radio.connect(addr, timeout=self.timeout)
        # sending the actual message
        conn[UARTService].write(class_name.lower().encode('ascii'))
        # disconnecting from the controller
        conn.disconnect()
        # I keep the controllers disconnected from this system in case I would like to trigger switches from my phone


'''
manager = BLEConnectionManager()
while True:
    """
    manager.send_message("theater")
    sleep(4)
    manager.send_message("bar")
    sleep(4)
'''

