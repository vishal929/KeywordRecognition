# logic for sending messages to specific microcontrollers and connecting to them
from threading import Thread
from multiprocessing import Process,Lock
from time import sleep, time

from adafruit_ble import BLERadio
from PiSystem.constants import SWITCH_DEVICE_MAP,UART_SERVICE_UUID,UART_RX_CHAR_UUID
from adafruit_ble.services.nordic import UARTService
from bleak import BleakScanner,BLEDevice,AdvertisementData,BleakClient
from bleak.backends.characteristic import BleakGATTCharacteristic
import asyncio

'''
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
        try:
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
                return
            device = SWITCH_DEVICE_MAP[class_name]
            addr = self.discover_device(device)
            if addr is None:
                print(' could not discover device: ' + str(device) +' -> will abort message sending...')
                return
            conn = self.radio.connect(addr, timeout=self.timeout)
            # sending the actual message
            conn[UARTService].write(class_name.lower().encode('ascii'))
            # disconnecting from the controller
            conn.disconnect()
        except Exception as e:
            print('exception occured while trying to send message: ' + str(class_name))
            print('exception string: ' + str(e))
        finally:
            return
'''
'''
manager = BLEConnectionManager()
while True:
    manager.send_message("theater")
    sleep(4)
    manager.send_message("bar")
    sleep(4)
    manager.send_message(' b;;;; a ;;;; r;;; ')
'''


async def send_message_async(class_tag):
    #stop_event = asyncio.Event()
    # handling string cleaning
    filtered = ""
    for c in class_tag:
        if c.isalpha():
            filtered += c.lower()
    class_tag = filtered
    if class_tag not in SWITCH_DEVICE_MAP:
        # invalid tag given we abort
        print('invalid tag given: ' + str(class_tag))
        #for task in asyncio.all_tasks():
        #task.cancel()
        return

    device_name = SWITCH_DEVICE_MAP[class_tag]
    # TODO: add something that calls stop_event.set()

    '''
    def device_filter(device: BLEDevice, adv: AdvertisementData):
        if device.name is not None and device_name in device.name:
            # we have the device
            return True
        return False
    '''



    device = await BleakScanner.find_device_by_name(device_name,timeout=10)
    if device is None:
        print('failed to find the device')
        return
    await BleakScanner.stop()

    print(device)

    def handle_disconnect(_: BleakClient):
        print("Device was disconnected, goodbye.")

    async with BleakClient(device, handle_disconnect) as client:
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
        await client.write_gatt_char(rx_char, class_tag.encode('ascii'))
        await client.disconnect()
    await BleakScanner.stop()


    '''
    async def connection_callback(device: BLEDevice, adv: AdvertisementData):
        curr_time = time()
        if curr_time-initial_time>10:
            print('connection timeout...')
            stop_event.set()
        if device.name is None:
            return
        if device not in devices:
            devices.add(device)
            if device_name in device.name:
                print(device.name)
                async with BleakClient(device, handle_disconnect) as client:
                    nus = client.services.get_service(UART_SERVICE_UUID)
                    rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
                    await client.write_gatt_char(rx_char, class_tag.encode('ascii'))
                    # issue the stop event
                    stop_event.set()
    

    async with BleakClient(device,handle_disconnect) as client:
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
        await client.write_gatt_char(rx_char,class_tag.encode('ascii'))

    def callback(device, advertising_data):
        if device.name is not None and device_name in device.name:
            # we have the device, lets send the message
            stop_event.set()
    async with BleakScanner(connection_callback) as scanner:
        ...
        # Important! Wait for an event to trigger stop, otherwise scanner
        # will stop immediately.
        await stop_event.wait()
    '''

def send_message(class_tag,ble_mutex):
    ble_mutex.acquire()
    asyncio.run(send_message_async(class_tag))
    ble_mutex.release()

if __name__ == '__main__':
    # testing
    ble_mutex = Lock()
    p = Process(target=send_message, args=('bar',ble_mutex))
    p2 = Process(target=send_message, args=('theater',ble_mutex))
    p.start()
    p2.start()

    p.join()
    p2.join()