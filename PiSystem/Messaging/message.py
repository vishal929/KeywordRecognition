# logic for sending messages to specific microcontrollers and connecting to them
from adafruit_ble import BLERadio, BLEConnection
from PiSystem.constants import INV_MAP
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
        # we should have named connections of the form (connection_object, device_name)
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
        self.ble_connections is updated with the established connections and device names
        :param device_addrs: This is a list of the form (bluetooth_address, device_complete_name)
        :param timeout: This is the number of seconds until a connection attempt is considered failed
    """
    def connect_to_devices(self, device_addrs, timeout=10):
        for addr, name in device_addrs:
            conn = self.radio.connect(addr, timeout=timeout)
            # pairing attempt
            #conn.pair()
            self.ble_connections.append((conn, name))

    """ 
        sending a message via UART to a microcontroller in our device list.
        Device complete name + class_trigger uniquely identify the microcontroller to send the message to.
        :param class_trigger: an integer representing the class to trigger ( this is the message to send)
    """
    def send_message(self, class_trigger):
        class_name = INV_MAP[class_trigger]
        for conn, name in self.ble_connections:
            if class_name in name.lower():
                conn[UARTService].write(str(class_trigger).encode('ascii'))
                return


#manager = BLEConnectionManager()
#manager.send_message(4)

