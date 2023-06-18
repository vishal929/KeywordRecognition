# holds logic for listening to messages from ble devices to act as a central
# this code is based on the rfcomm server example code in pybluez git repo

from bluetooth import BluetoothSocket,PORT_ANY,RFCOMM,SERIAL_PORT_PROFILE,SERIAL_PORT_CLASS,advertise_service

server_sock = BluetoothSocket(RFCOMM)
server_sock.bind(("", PORT_ANY))
server_sock.listen(1)

port = server_sock.getsockname()[1]

uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
rx_uuid = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
tx_uuid = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
#uuid = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"

advertise_service(server_sock, "SampleServer", service_id=uuid,
                            service_classes=[rx_uuid],
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



