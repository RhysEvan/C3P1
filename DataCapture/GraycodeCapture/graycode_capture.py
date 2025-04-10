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
        for i, camera in enumerate(cameras):

            frame = camera.GetFrame()
            frame = frame.astype(np.uint8)

            if frame.ndim == 2:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                bgr_frame[frame <= 5] = [255, 0, 0]
                bgr_frame[frame >= 250] = [0, 0, 255]
            elif frame.ndim == 3:
                bgr_frame = frame
            else:
                raise ValueError("Unsupported frame format")

            bgr_frame = cv2.resize(bgr_frame, dimensions[i], interpolation=cv2.INTER_AREA)

            window_name = 'Camera Feed' if len(cameras) == 1 else f'Camera Feed {i+1}'
            cv2.imshow(window_name, bgr_frame)

        if cv2.waitKey(1) & 0xFF == 13:
            stop_event.set()
            break

    cv2.destroyAllWindows()

def pygame_event_loop(stop_event):
    while not stop_event.is_set():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_event.set()
                break
        time.sleep(0.01)  # Small delay to reduce CPU usage

def get_dataset_name(base_name):
    base_name = base_name.rsplit('.', 1)[0]
    parts = base_name.split('_')
    if parts[-1].isdigit():
        parts = parts[:-1]
    if len(parts) > 2:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    elif len(parts) > 1:
        return f"{parts[0]}_{parts[1]}"
    return base_name

def capture_feed(screen, file_path, cameras, h5_file, gamma = False):
    if len(cameras) == 0:
        print('something is wrong, no cameras were given')
        return
    pygame.event.get()
    texture_frame = None
    key = ""

    for _, img_path in enumerate(glob.glob(os.path.join(file_path, '*'))):
        if "tiff" not in img_path:
            continue
        else:
            base_name = os.path.basename(img_path)
            dataset_name = get_dataset_name(base_name)

            imgToScrn(screen, img_path)
            time.sleep(0.16)

            for i, camera in enumerate(cameras):
                if camera == cameras[-1]:
                    continue

                frame = camera.GetFrame()
                
                if frame.dtype != np.uint8 and gamma == False:
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

                if "white" in dataset_name and i == 1:
                    rgb_key = "RGB_white"
                    texture_frame = cameras[-1].GetFrame()
                else:
                    rgb_key = None

                if key not in h5_file:
                    initial_shape = (0, 1536, 2048)
                    max_shape = (None, 1536, 2048)
                    chunks = (1, 1536, 2048)
                    h5_file.create_dataset(key, shape=initial_shape,
                                           maxshape=max_shape, chunks=chunks, dtype=np.uint8)
                dataset = h5_file[key]
                dataset.resize((dataset.shape[0] + 1, 1536, 2048))
                dataset[-1, :, :] = frame
                
                if rgb_key is not None:
                    if rgb_key not in h5_file:
                        shape = texture_frame.shape
                        initial_shape = (0, shape[0], shape[1], shape[2])
                        max_shape = (None, shape[0], shape[1], shape[2])
                        chunks = (1, shape[0], shape[1], shape[2])
                        h5_file.create_dataset(rgb_key, shape=initial_shape,
                                            maxshape=max_shape, chunks=chunks, dtype=np.uint8)

                    rgb_dataset = h5_file[rgb_key]
                    rgb_dataset.resize((rgb_dataset.shape[0] + 1, shape[0], shape[1], shape[2]))
                    rgb_dataset[-1, :, :, :] = texture_frame

def turntable_capture(screen, file_path, cameras, store_path,
                      range_steps, serial_port='COM7', baudrate=115200):
    """
    Rotates the motor in equal steps based on the specified range.
    
    :param range_steps: The number of steps to divide the 360-degree rotation into.
    :param serial_port: The serial port to which the Arduino is connected.
    :param baudrate: The baud rate for serial communication.
    """

    ser = serial.Serial(serial_port, baudrate, timeout=1)
    print(ser)
    print("Serial connection established")
    time.sleep(2)

    for j in range(range_steps):
        h5_path = create_incremented_h5(store_path, "scan")
        with h5py.File(h5_path, 'a') as h5_file:
            angle_step = 360 / range_steps
            capture_feed(screen, file_path, cameras, h5_file)
            angle = angle_step * (j+1)
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

def graycode_data_capture(file_path, cameras, dimensions, store_path, range_steps, gamma = False):
    pygame.init()
    display_info = pygame.display.Info()
    num_displays = pygame.display.get_num_displays()

    img = pygame.image.load(file_path+"pattern_white.tiff")
    siRect = img.get_rect()
    if num_displays < 2:
        screen = pygame.display.set_mode((siRect[2], siRect[3]), pygame.FULLSCREEN, display=0)
    else:
        screen = pygame.display.set_mode((siRect[2], siRect[3]), pygame.FULLSCREEN, display=1)

    screen.blit(img, siRect)
    pygame.display.flip()
    time.sleep(1)

    # Create a stop event for the camera feed thread
    stop_event = Event()

    # Start Pygame event handling thread
    thread_pygame_events = Thread(target=pygame_event_loop, args=(stop_event,))
    thread_pygame_events.start()

    # Start camera feed thread
    thread_camera_feed = Thread(target=show_camera_feed, args=(cameras, dimensions, stop_event))
    thread_camera_feed.start()

    thread_camera_feed.join()
    thread_pygame_events.join()

    # Open HDF5 file
    if range_steps < 2:
        h5_path = create_incremented_h5(store_path, "scan")
        with h5py.File(h5_path, 'a') as h5_file:
            # Start the image feed thread
            capture_feed(screen, file_path, cameras, h5_file, gamma)
    else:
        turntable_capture(screen, file_path, cameras, store_path,
                          range_steps, 'COM3', 115200)

def imgToScrn(scrn, image_path):
    img = pygame.image.load(image_path)
    siRect = img.get_rect()
    scrn.blit(img, siRect)
    pygame.display.flip()

def create_incremented_h5(store_path, base_name):
    counter = 1
    while True:
        h5_path = os.path.join(store_path,
                               f"{base_name}_{counter}.h5")
        if not os.path.exists(h5_path):
            return h5_path
        counter += 1
