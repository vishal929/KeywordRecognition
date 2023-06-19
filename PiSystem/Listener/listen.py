# holds logic for listening to messages from ble devices to act as a central
# this code is based on the rfcomm server example code in pybluez git repo

# below import is for serial option
#from bluetooth import BluetoothSocket,PORT_ANY,RFCOMM,SERIAL_PORT_PROFILE,SERIAL_PORT_CLASS,advertise_service
import asyncio
from threading import Thread

from bluez_peripheral.gatt.service import Service
from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags as CharFlags
from bluez_peripheral.util import *
from bluez_peripheral.advert import Advertisement
from bluez_peripheral.agent import NoIoAgent
from bluez_peripheral.gatt.descriptor import descriptor,DescriptorFlags as DescFlags

from multiprocessing import Process

from PiSystem.Messaging.message import BLEConnectionManager

rx_uuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
tx_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
desc1_uuid = "2901"
desc2_uuid = "2902"
# service for listening to phone messages as a peripheral for BLE
class ListenerService(Service):
    # we pass in our message handler
    def __init__(self):
        self.message = None
        # service UUID for nordic ble uart service (just to keep it consistent)
        super().__init__(uuid,True)

    # TX characteristic of UART service
    @characteristic(tx_uuid,CharFlags.NOTIFY)
    def read_message(self,options):
        return

    ''' 
    @descriptor(desc2_uuid,DescFlags.READ)
    def read_message_desc2(self,options):
        return
    '''

    @descriptor(desc1_uuid,read_message,DescFlags.READ | DescFlags.WRITE)
    def read_message_desc1(self,value,options):
        return bytes("TXD","utf-8")

    @characteristic(rx_uuid,CharFlags.WRITE|CharFlags.WRITE_WITHOUT_RESPONSE).setter
    def set_message(self,value,options):
        # trigger our message handler
        self.message = value.decode('utf-8')

    @descriptor(desc1_uuid,set_message,DescFlags.WRITE | DescFlags.READ)
    def set_message_desc1(self,options):
        return

class ListenThread(Process):

    def __init__(self, queue):
        super().__init__()
        # the queue here will have a reference to the connection manager to use
        self.connection_manager = BLEConnectionManager()
        # we will have our own connection manager but a shared mutex for ble requests
        self.ble_mutex = queue.get_nowait()
    async def start_ble_logic(self):
       # Alternatively you can request this bus directly from dbus_next.
       bus = await get_message_bus()
       service = ListenerService()
       await service.register(bus)

       # An agent is required to handle pairing
       agent = NoIoAgent()
       # This script needs superuser for this to work.
       await agent.register(bus)

       adapter = await Adapter.get_first(bus)

       # Start an advert that will last for 60 seconds.
       advert = Advertisement("raspberry_pi_listener", [uuid], 0, 0)
       await advert.register(bus, adapter)

       while True:
           if service.message is not None:
               self.ble_mutex.acquire()
               self.connection_manager.send_message(service.message)
               # resetting the message
               service.message = None
               self.ble_mutex.release()
           # Handle dbus requests.
           await asyncio.sleep(5)

       await bus.wait_for_disconnect()


    def run(self):
       asyncio.run(self.start_ble_logic())

'''
IN CASE YOU WANT TO USE SERIAL OR YOUR DEVICE DOES NOT SUPPORT LOW ENERGY BLUETOOTH
YOU CAN USE THE BELOW CODE WITH PYBLUEZ LIBRARY TO ACCEPT STANDARD BLUETOOTH MESSAGES IN SERIAL
server_sock = BluetoothSocket(RFCOMM)
server_sock.bind(("", PORT_ANY))
server_sock.listen(1)


port = server_sock.getsockname()[1]

uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
#rx_uuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
#tx_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"

advertise_service(server_sock, "SampleServer", service_id=uuid,
                            service_classes=[uuid,SERIAL_PORT_CLASS],
                            #profiles=[SERIAL_PORT_PROFILE],
                            # protocols=[OBEX_UUID]
                            )

print("Waiting for connection on RFCOMM channel", port)

client_sock, client_info = server_sock.accept()
print("Accepted connection from", client_info)

try:
    while True:
        data = client_sock.recv(1024)
        if not data:
            break
        print("Received", data)
except OSError:
    pass

print("Disconnected.")

client_sock.close()
server_sock.close()
print("All done.")
'''

