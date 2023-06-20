# logic for sending messages to specific microcontrollers and connecting to them
import asyncio
from multiprocessing import Process, Lock

from bleak import BleakScanner, BleakClient

from PiSystem.constants import SWITCH_DEVICE_MAP, UART_SERVICE_UUID, UART_RX_CHAR_UUID


async def send_message_async(class_tag):
    """
    Async function which handles scanning for the device, connecting, and sending a message.
    In case of any failure, we print the failure and return, since this async function should be
    wrapped to be called in a separate thread/process


    :parameter class_tag: This should be the name of the switch to trigger i.e "stairs","bar",etc.
    The valid class tags can be viewed in PiSystem/constants.py

    """
    # handling string cleaning
    filtered = ""
    for c in class_tag:
        if c.isalpha():
            filtered += c.lower()
    class_tag = filtered
    if class_tag not in SWITCH_DEVICE_MAP:
        # invalid tag given we abort
        print('invalid tag given: ' + str(class_tag))
        return

    device_name = SWITCH_DEVICE_MAP[class_tag]

    device = await BleakScanner.find_device_by_name(device_name, timeout=10)
    if device is None:
        print('failed to find the device')
        return

    print(device)

    def handle_disconnect(_: BleakClient):
        print("Device was disconnected, goodbye.")

    async with BleakClient(device, handle_disconnect) as client:
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)
        await client.write_gatt_char(rx_char, class_tag.encode('ascii'))
        await client.disconnect()


def send_message(class_tag, ble_mutex):
    """
    This is the synchronous wrapper around the send_message_async function.
    We need to acquire a mutex in order to prevent deadlock which can occur when
    connecting multiple times in BLE.

    :param class_tag:
        This should be the name of the switch to trigger, like "bar","stairs",etc.
        The full list of class_tags is given in PiSystem/constants.py
    :param ble_mutex:
        This is a multiprocessing lock. We need to use locking mechanisms in order to prevent
        deadlock when connecting to devices in BLE, since scanning multiple times causes issues.
    :return: void
    """
    ble_mutex.acquire()
    try:
        asyncio.run(send_message_async(class_tag))
    except Exception as e:
        print('exception while sending message: ' + str(e))
    ble_mutex.release()


if __name__ == '__main__':
    # testing
    ble_mutex = Lock()
    p = Process(target=send_message, args=('bar', ble_mutex))
    p2 = Process(target=send_message, args=('theater', ble_mutex))
    p.start()
    p2.start()

    p.join()
    p2.join()
