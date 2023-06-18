# holds logic for listening to messages from ble devices to act as a central
# this code is based on the rfcomm server example code in pybluez git repo

from bluetooth import BluetoothSocket,PORT_ANY,RFCOMM,SERIAL_PORT_PROFILE,SERIAL_PORT_CLASS,advertise_service
from bluez_peripheral.gatt.service import Service
from bluez_peripheral.gatt.characteristic import characteristic, CharacteristicFlags as CharFlags
from bluez_peripheral.util import *
from bluez_peripheral.advert import Advertisement
from bluez_peripheral.agent import NoIoAgent
import asyncio

rx_uuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
tx_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
# service for listening to phone messages as a peripheral for BLE
class ListenerService(Service):
    def __init__(self):
        # message that phones will write to
        self.message = None
        # service UUID for nordic ble uart service (just to keep it consistent)
        super().__init__(uuid,True)

    # RX characteristic of UART service
    @characteristic(rx_uuid,CharFlags.WRITE_WITHOUT_RESPONSE)
    def read_message(self,value,options):
        self.message = value.decode('ascii')

async def main():
    # Alternativly you can request this bus directly from dbus_next.
    bus = await get_message_bus()

    service = ListenerService()
    await service.register(bus)

    # An agent is required to handle pairing
    agent = NoIoAgent()
    # This script needs superuser for this to work.
    await agent.register(bus)

    adapter = await Adapter.get_first(bus)

    # Start an advert that will last for 60 seconds.
    advert = Advertisement("Vishal's Pi Listener", [uuid,rx_uuid], 0x0340, 0)
    await advert.register(bus, adapter)

    while True:
        print(service.message)
        # Handle dbus requests.
        await asyncio.sleep(5)

    await bus.wait_for_disconnect()

asyncio.run(main())
'''
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



