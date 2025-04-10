import pygame
import os
import cv2
import h5py
import serial
import time
import glob
import numpy as np

from threading import Thread, Event

def show_camera_feed(cameras, dimensions, stop_event):
    if len(cameras) == 0:
        print('something is wrong, no cameras were given')
        return

    while not stop_event.is_set():
        # Capture and display frames from cameras
        for i, camera in enumerate(cameras):

            frame = camera.GetFrame()
            frame = frame.astype(np.uint8)

            # Convert to BGR for display
            if frame.ndim == 2:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                bgr_frame[frame <= 5] = [255, 0, 0]
                bgr_frame[frame >= 250] = [0, 0, 255]
            elif frame.ndim == 3:
                bgr_frame = frame
            else:
                raise ValueError("Unsupported frame format")

            # Resize to fit the specified dimensions
            bgr_frame = cv2.resize(bgr_frame, dimensions[i], interpolation=cv2.INTER_AREA)

            # Show in OpenCV window
            window_name = 'Camera Feed' if len(cameras) == 1 else f'Camera Feed {i+1}'
            cv2.imshow(window_name, bgr_frame)

        # OpenCV waits for a short period, allows Pygame events to be processed
        if cv2.waitKey(1) & 0xFF == 13:  # Break loop on Enter key
            stop_event.set()
            break

    cv2.destroyAllWindows()

def capture_feed(angle, cameras, h5_file):
    if len(cameras) == 0:
        print('something is wrong, no cameras were given')
        return

    dataset_name = str(angle)

    for i, camera in enumerate(cameras):
        if len(cameras) == 3:
            if camera == cameras[-1]:
                continue
        frame = camera.GetFrame()
        if frame.dtype != np.uint8:
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = (frame * 255).astype(np.uint8)
            frame = (frame).astype(np.uint8)
        else:
            frame = (frame).astype(np.uint8)

        if len(cameras) == 1:
            key = f"{dataset_name}"
        elif len(cameras) == 3 and i == 0:
            key = f"L_{dataset_name}"
        elif len(cameras) == 3 and i == 1:
            key = f"R_{dataset_name}"

        if key not in h5_file:
            initial_shape = (0, 1536, 2048)
            max_shape = (None, 1536, 2048)
            chunks = (1, 1536, 2048)
            h5_file.create_dataset(key, shape=initial_shape,
                                   maxshape=max_shape, chunks=chunks, dtype=np.uint8)
        dataset = h5_file[key]
        dataset.resize((dataset.shape[0] + 1, 1536, 2048))
        dataset[-1, :, :] = frame

def turntable_calibration_capture(cameras, store_path,
                                  range_steps, serial_port='COM7', baudrate=115200):
    """
    Rotates the motor in equal steps based on the specified range.

    :param range_steps: The number of steps to divide the 360-degree rotation into.
    :param serial_port: The serial port to which the Arduino is connected.
    :param baudrate: The baud rate for serial communication.
    """

    ser = serial.Serial(serial_port, baudrate, timeout=1)  # Increased timeout to 1 second
    print(ser)
    print("Serial connection established")
    time.sleep(2)

    # Create a stop event for the camera feed thread
    stop_event = Event()

    # Start camera feed thread
    thread_camera_feed = Thread(target=show_camera_feed, args=(cameras, [(640, 480), (640, 480)], stop_event))
    thread_camera_feed.start()

    thread_camera_feed.join()

    h5_path = create_incremented_h5(store_path, "scan")
    with h5py.File(h5_path, 'a') as h5_file:
        for j in range(range_steps):
            angle_step = 360 / range_steps
            angle = angle_step * (j+1)
            capture_feed(angle, cameras, h5_file)
            print(f"Step {j + 1}/{range_steps}: Rotating to {angle} degrees")
            send_motor_command(ser, angle)
        ser.write("done\r\n".encode())
        ser.flush()

def send_motor_command(ser, angle):
    """
    Sends a command to rotate the motor to the specified angle and waits until the
    Arduino is ready for the next command.

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
            time.sleep(1)  # Small delay before reading the response

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

def create_incremented_h5(store_path, base_name):
    while True:
        h5_path = os.path.join(store_path,
                               f"{base_name}.h5")
        if not os.path.exists(h5_path):
            return h5_path