# holds logic for listening to messages from bluetooth serial devices
# the idea is that in addition to triggering switches with voice, I could also trigger via phone
# this code is based on the rfcomm server example code in pybluez git repo

from multiprocessing import Lock 
# below import is for serial option
from bluetooth import BluetoothSocket, PORT_ANY, RFCOMM, SERIAL_PORT_CLASS, advertise_service
from PiSystem.Messaging.message import send_message


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
        server_sock = BluetoothSocket(RFCOMM)
        server_sock.bind(("", PORT_ANY))
        server_sock.listen(1)

        port = server_sock.getsockname()[1]

        serial_uuid = "94f39d29-7d6d-437d-973b-fba39e49d4ee"
        advertise_service(server_sock, "SampleServer", service_id=serial_uuid,
                          service_classes=[serial_uuid, SERIAL_PORT_CLASS],
                          )

        if os.fork()==0:
            # 1 child process will listen
            handle_message("child")

        # the parent will also listen
        handle_message("parent")

    def handle_message(process_id):
        """ 
        Handling clients
        """
        while True:
            print("Waiting for connection on RFCOMM channel", port)

            client_sock, client_info = server_sock.accept()
            print("Accepted connection from ", client_info, " in " + process_id)

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
            print("All done.")


if __name__ == "__main__":
    # testing
    lock = Lock()
    proc = BluetoothListener(lock)
    proc.start()
    proc.join()
