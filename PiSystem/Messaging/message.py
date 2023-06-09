# logic for sending messages to specific microcontrollers and connecting to them
from adafruit_ble import BLERadio,BLEConnection
from PiSystem.constants import INV_MAP
from adafruit_ble.services.nordic import UARTService

# this function serves to discover microcontrollers by bluetooth name
# in our case since we are dealing with the Adafruit itsybitsy nrf52840 our keyword is "ItsyBitsy"
# we are only going to scan for 30s before timing out
def discover_devices(keyword = 'ItsyBitsy', timeout=30):
    radio = BLERadio()
    found = set()
    desired = []
    for entry in radio.start_scan(timeout=timeout):
        addr = entry.address
        if addr not in found:
            if entry.complete_name and keyword in entry.complete_name:
                print('complete name: '+ str(entry.complete_name))
                desired.append((addr,  entry.complete_name))
            found.add(addr)
    return radio,desired

# given a list of device addresses we want to connect to, lets try connecting to them
# we will timeout after 10 seconds if no connection
def connect_to_devices(radio,device_addrs,timeout=10):
    ble_connections = []
    for addr,name in device_addrs:
        ble_connections.append(
            (radio.connect(addr,timeout=timeout),name)
        )
    return ble_connections

# sending a message to a microcontroller
# ble_connections is our list of paired microcontrollers in the form (connection, name)
# class_trigger is the integer representing the class to trigger
def send_message(ble_connections, class_trigger):
    class_name = INV_MAP[class_trigger]
    for conn, name in ble_connections:
        if class_name in name.lower():
            conn[UARTService].write(class_trigger.to_bytes(length=1,byteorder='big'))
            return


radio,desired = discover_devices(timeout=5)
ble_connections = connect_to_devices(radio,desired)
send_message(ble_connections,1)