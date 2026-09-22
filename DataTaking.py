import configparser
from pathlib import Path

import serial
import time
import tkinter as tk
import tkinter.font as tkFont


CONFIG_FILE = Path(__file__).resolve().parent / "config.ini"
config = configparser.ConfigParser()
config.read(CONFIG_FILE)
dev_dir = config.get("detector", "serial_port", fallback="/dev/tty.usbserial-0001")

def get_scalar(str_val, i):
    """
    Extracts a scalar value from a string response.
    str_val: The string containing the response.
    i: The scalar index (e.g., '0', '1', '4').
    """
    S_name = 'S' + i
    words = str_val.split(' ')
    for word in words:
        if S_name in word:
            # Extract the value after 'S_name='
            value_str = word.replace(S_name + '=', '')
            try:
                # Return as integer from hex string
                return int(value_str, 16)
            except ValueError:
                print(f"Could not convert {value_str} to int")
                return 0
    return 0

def count():
    """
    Placeholder function for a timer or counter update.
    """
    time.sleep(1)
    print('COUNTED')
    # Assuming 'txt' is a global Label widget
    global txt
    txt['text'] = '%%'
    return 0

def countD(duration, scalar):
    """
    Opens a serial connection, sends commands, and reads counter values.
    """
    # open connection
    try:
        #ser = serial.Serial('/dev/tty.SLAB_USBtoUART', baudrate=115200, xonxoff=1, timeout=1)
        ser = serial.Serial(dev_dir, baudrate=115200, xonxoff=1, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

    try:
        # send commands
        ser.write(b"CD\r")  # disable triggers display
        start_time = time.localtime()
        
        ser.write(b"RB\r")  # reset counters
        time.sleep(duration)
        
        stop_time = time.localtime()
        
        ser.write(b"DS\r")  # display counters
        time.sleep(1)
        
        response = ser.read(500).decode('utf-8', errors='ignore')  # Decode bytes to string
        #print(response)
        
        S0 = get_scalar(response, scalar)
        S1 = get_scalar(response, '0')
        S2 = get_scalar(response, '1')
        S4 = get_scalar(response, '4')
        
        # Close the serial connection
        ser.close()
        
        return {
            'C1': S1,
            'C2': S2,
            'COINC': S4
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        ser.close()
        return None

# Example usage (if you want to test it outside of a GUI)
if __name__ == "__main__":
    # Initialize a dummy root and label for the 'txt' global variable
    #root = tk.Tk()
    #root.withdraw()  # Hide the main window
    #txt = tk.Label(root, text="")
    #txt.pack()
    
    # Test the count_ function (adjust duration and scalar as needed)
    # Note: This will fail if no device is connected at /dev/tty.SLAB_USBtoUART
    duration = 10
    result = countD(duration, '0')
    if result:
        print(f"Results: {result}")
