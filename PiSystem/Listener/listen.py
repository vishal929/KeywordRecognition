# holds logic for listening to messages from bluetooth serial devices
# the idea is that in addition to triggering switches with voice, I could also trigger via phone
# this code is based on the rfcomm server example code in pybluez git repo

from multiprocessing import Lock 
# below import is for serial option
#from bluetooth import BluetoothSocket, PORT_ANY, RFCOMM, SERIAL_PORT_CLASS, advertise_service
import socket
from PiSystem.Messaging.message import send_message
import os

class BluetoothListener():
    """
        BluetoothListener is a process that listens for incoming messages via bluetooth.
        This allows me to communicate with the raspberry pi through bluetooth from my phone,
        and so I can activate switches that way
    """

    def __init__(self, mutex):
        """
        Constructor for our BluetoothListener

        :param mutex:
            We require a mutex to avoid deadlock in the messaging service.
            From testing I found that connecting to a device sometimes starts a scan.
            With multiple scans, we may run into BLE deadlock, so we acquire a mutex for this purpose.
        """
        super().__init__()
        self.mutex = mutex

    def listen(self):
        """
        Runnable for our bluetooth listener service
        We continually accept 1 client connection on the bluetooth socket and listen to messages.
        :return: void
        """
        b_adapter = 'dc:a6:32:85:26:96'
        port = 3
        s = socket.socket(socket.AF_BLUETOOTH,socket.SOCK_STREAM,socket.BTPROTO_RFCOMM)
        s.bind((b_adapter,port))
        s.listen(1)
        self.handle_message(s,port)


    def handle_message(self,sock):
        """ 
        Handling clients
        """
        while True:
            print("Waiting for connection on RFCOMM channel", port)

            client_sock, client_info = sock.accept()
            print("Accepted connection from ", client_info)

            try:
                while True:
                    message = client_sock.recv(1024)
                    if not message:
                        break
                    # sending the message in a new process
                    Process(target=send_message, args=(message.decode('ascii'), self.mutex)).start()
                    #send_message(message.decode('ascii'),self.mutex)
            except OSError:
                pass

            client_sock.close()
            print("All done.")


if __name__ == "__main__":
    # testing
    lock = Lock()
    proc = BluetoothListener(lock)
    proc.start()
    proc.join()
