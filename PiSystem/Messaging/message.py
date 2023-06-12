# logic for sending messages to specific microcontrollers and connecting to them
from time import sleep

from adafruit_ble import BLERadio
from PiSystem.constants import LEARN_MAP
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
        # we should have named connections of the form (connection_object, device_addr, device_name)
        self.ble_connections = []
        # we should set a timeout of seconds until the connection is considered failed
        self.timeout = timeout
        # lets initialize our connections
        discovered = self.discover_devices(timeout=self.timeout)
        self.connect_to_devices(discovered, timeout=self.timeout)

    """ 
        This function serves to discover microcontrollers by bluetooth name
        In our case since we are dealing with the Adafruit itsybitsy nrf52840 our keyword is "ItsyBitsy"
        :param keyword: This is a string which should be in device complete names of devices we need to connect to
        :param timeout: This is the number of seconds until a connection attempt is considered failed
        :return desired: This is a list of the format [(bluetooth_address, device_complete_name)]
    """
    def discover_devices(self, keyword='ItsyBitsy', timeout=30):
        found = set()
        desired = []
        for entry in self.radio.start_scan(timeout=timeout):
            addr = entry.address
            if addr not in found:
                if entry.complete_name and keyword in entry.complete_name:
                    print('complete name: ' + str(entry.complete_name))
                    desired.append((addr, entry.complete_name))
                found.add(addr)
        return desired

    # given a list of device addresses we want to connect to, lets try connecting to them
    # we will timeout after 10 seconds if no connection
    """
        Using BLE to actually establish connections to a list of device addresses
        self.ble_connections is updated with the established connections, addresses, and device names
        :param device_addrs: This is a list of the form (bluetooth_address, device_complete_name)
        :param timeout: This is the number of seconds until a connection attempt is considered failed
    """
    def connect_to_devices(self, device_addrs, timeout=10):
        for addr, name in device_addrs:
            conn = self.radio.connect(addr, timeout=timeout)
            # pairing attempt
            #conn.pair()
            self.ble_connections.append((conn, addr, name))

    """ 
        sending a message via UART to a microcontroller in our device list.
        Device complete name + class_trigger uniquely identify the microcontroller to send the message to.
        :param class_trigger: class name of the switch to trigger (i.e bar, theater, etc. defined in constants.py)
    """
    def send_message(self, class_trigger):
        class_int = LEARN_MAP[class_trigger]
        for idx, (conn, addr, name) in enumerate(self.ble_connections):
            # check if we are connected
            if class_trigger in name.lower():
                if conn.connected:
                    conn[UARTService].write(str(class_int).encode('utf-8'))
                    return
                else:
                    print('we are not connected, will attempt to reconnect...')
                    devices = self.discover_devices(keyword=name,timeout=5)
                    if len(devices) == 0:
                        print('reconnecting to : ' + name + ' failed...')
                        return
                    # we have data, so get the first index
                    # device is a tuple of (addr, device_name)
                    device = devices[0]
                    # reestablishing connection
                    new_conn = self.radio.connect(device[0], timeout=5)
                    # checking if the new connection is valid
                    if new_conn.connected:
                        print('successfully reconnected to: ' + device[1])
                        # updating the list
                        self.ble_connections[idx] = (new_conn,device[0],device[1])
                        # doing the write
                        new_conn[UARTService].write(str(class_int).encode('utf-8'))
                        return

"""
manager = BLEConnectionManager()
while True:
    manager.send_message(4)
    sleep(4)
"""

