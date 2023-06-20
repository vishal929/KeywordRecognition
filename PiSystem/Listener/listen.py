# holds logic for listening to messages from bluetooth serial devices
# the idea is that in addition to triggering switches with voice, I could also trigger via phone
# this code is based on the rfcomm server example code in pybluez git repo

from multiprocessing import Lock, Process
# below import is for serial option
from bluetooth import BluetoothSocket, PORT_ANY, RFCOMM, SERIAL_PORT_CLASS, advertise_service
from PiSystem.Messaging.message import send_message


class BluetoothListener(Process):
    """
        BluetoothListener is a process that listens for incoming messages via bluetooth.
        This allows me to communicate with the raspberry pi through bluetooth from my phone,
        and so I can activate switches that way
    """
    def __init__(self, mutex):
        super().__init__()
        self.server_sock = BluetoothSocket(RFCOMM)
        self.server_sock.bind(("", PORT_ANY))
        self.server_sock.listen(1)

        self.port = self.server_sock.getsockname()[1]

        self.serial_uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
        self.mutex = mutex

    def run(self) -> None:
        advertise_service(self.server_sock, "SampleServer", service_id=self.serial_uuid,
                          service_classes=[self.serial_uuid, SERIAL_PORT_CLASS],
                          )

        print("Waiting for connection on RFCOMM channel", self.port)

        client_sock, client_info = self.server_sock.accept()
        print("Accepted connection from", client_info)

        try:
            while True:
                message = client_sock.recv(1024)
                if not message:
                    break
                # sending the message in a new process
                Process(target=send_message, args=(message.decode('ascii'), self.mutex)).start()
        except OSError:
            pass

        client_sock.close()
        self.server_sock.close()
        print("All done.")


if __name__ == "__main__":
    # testing
    lock = Lock()
    proc = BluetoothListener(lock)
    proc.start()
    proc.join()
