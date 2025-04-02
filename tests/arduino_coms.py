import serial
import time

def send_motor_command(ser, angle):
    """
    Sends a command to rotate the motor to the specified angle and waits until the Arduino is ready for the next command.

    :param angle: The angle to rotate the motor.
    :param serial_port: The serial port to which the Arduino is connected.
    :param baudrate: The baud rate for serial communication.
    """
    try:
        # Send the motor command
        command = f"M 0 {angle}\r\n"
        ser.write(command.encode())
        ser.flush()
        print(f"Sent command: {command.strip()}")
        time.sleep(2)

        # Wait for the Arduino to be ready for the next command
        while True:
            # Send the "ready" command
            ser.write(b'ready\r\n')
            time.sleep(0.5)  # Small delay before reading the response

            # Read the response from the Arduino
            response = ser.readline().decode().strip()

            if response == "Ready for next command":
                print(response)
                break
            elif response == "Busy":
                print("Busy")
            else:
                print("Waiting for readiness...")
            time.sleep(0.5)  # Small delay before retrying

    except serial.SerialException as e:
        print(f"Error: {e}")

def rotate_motor_in_steps(range_steps, serial_port='COM7', baudrate=115200):
    """
    Rotates the motor in equal steps based on the specified range.
    
    :param range_steps: The number of steps to divide the 360-degree rotation into.
    :param serial_port: The serial port to which the Arduino is connected.
    :param baudrate: The baud rate for serial communication.
    """
    angle_step = 360 / range_steps

    ser = serial.Serial(serial_port, baudrate, timeout=1)  # Increased timeout to 1 second
    print(ser)
    print("Serial connection established")
    time.sleep(2)

    for i in range(range_steps):
        angle = angle_step * (i+1)
        print(f"Step {i + 1}/{range_steps}: Rotating to {angle} degrees")
        send_motor_command(ser, angle)

# Example usage:
range_steps = 10  # This means 10 steps, each 36 degrees
rotate_motor_in_steps(range_steps)
