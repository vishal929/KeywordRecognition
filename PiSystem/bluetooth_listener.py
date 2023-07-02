# this file houses logic for handling up to 2 connections via bluetooth
# this is in contrast to supporting voice recognition
# I found voice recognition to be a little shaky
# so, instead I will just opt to listen via bluetooth sockets only

# in the future, if I decide I want to continue with voice recognition,
# I will probably need a button or message to trigger the recording session
# this is harder than just sending a message via bluetooth anyway


from threading import Lock

from PiSystem.Listener.listen import BluetoothListener

if __name__ == '__main__':
    # we need a lock for ble connections
    ble_lock = Lock()
    # creating our connection handler for main thread
    listener = BluetoothListener(ble_lock)
    
    listener.listen()
