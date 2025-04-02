import os
import cv2
import h5py
import glob
import numpy as np

def show_camera_feed(camera, dimensions):
    window_name = 'Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = camera.GetFrame()
        if frame is None:
            print("Frame not captured correctly.")
            break
        frame = frame.astype(np.uint8)

        # Convert to BGR for display
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        bgr_frame[frame <= 5] = [255, 0, 0]
        bgr_frame[frame >= 250] = [0, 0, 255]

        # Resize to fit the specified dimensions
        bgr_frame = cv2.resize(bgr_frame, dimensions, interpolation=cv2.INTER_AREA)

        # Show in OpenCV window
        cv2.imshow(window_name, bgr_frame)

        if cv2.waitKey(10) & 0xFF == 13:  # Exit on Enter key
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def show_texture_feed(camera, dimensions):
    window_name = 'Texture Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Frame not captured correctly.")
            break
        frame = frame.astype(np.uint8)

        # Convert to BGR for display
        bgr_frame[frame <= 5] = [255, 0, 0]
        bgr_frame[frame >= 250] = [0, 0, 255]

        # Resize to fit the specified dimensions
        bgr_frame = cv2.resize(bgr_frame, dimensions, interpolation=cv2.INTER_AREA)

        # Show in OpenCV window
        cv2.imshow(window_name, bgr_frame)

        if cv2.waitKey(10) & 0xFF == 13:  # Exit on Enter key
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def intrinsic_calibration_capture(store_path, cameras, texture_camera = None, identifier=None):
    if len(cameras) == 0:
        print('Error: No cameras were provided.')
        return

    for i, camera in enumerate(cameras):
        if identifier is not None and len(identifier) == len(cameras):
            h5_path = create_incremented_h5(store_path, f"{identifier[i]}_scan")
        else:
            h5_path = create_incremented_h5(store_path, "scan")

        try:
            with h5py.File(h5_path, 'a') as h5_file:
                for j in range(40):
                    key = str(j)
                    show_camera_feed(camera, (640, 480))
                    frame = camera.GetFrame()
                    if frame is None:
                        print(f"Error: Frame {j} not captured.")
                        continue

                    if frame.dtype != np.uint8:
                        if frame.max() != frame.min():
                            frame = (frame - frame.min()) / (frame.max() - frame.min())
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = (frame * 255).astype(np.uint8)

                    if key not in h5_file:
                        initial_shape = (0, 1536, 2048)
                        max_shape = (None, 1536, 2048)
                        chunks = (1, 1536, 2048)
                        h5_file.create_dataset(key, shape=initial_shape,
                                               maxshape=max_shape, chunks=chunks, dtype=np.uint8)

                    dataset = h5_file[key]
                    dataset.resize((dataset.shape[0] + 1, 1536, 2048))
                    dataset[-1, :, :] = frame
        except Exception as e:
            print(f"Error accessing HDF5 file: {e}")
            return

    if len(texture_camera) == 0:
        print('Warning: No texture camera were provided.')
        return
    h5_path = create_incremented_h5(store_path, f"RGB_scan")
    for i in range(40):
        key = str(i)
        show_texture_feed(texture_camera, (640, 480))
        frame = texture_camera.GetFrame()


        if frame.dtype != np.uint8:
            if frame.max() != frame.min():
                frame = (frame - frame.min()) / (frame.max() - frame.min())
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = (frame * 255).astype(np.uint8)

        if key not in h5_file:
            shape = frame.shape
            initial_shape = (0, shape[0], shape[1])
            max_shape = (None, shape[0], shape[1])
            chunks = shape
            h5_file.create_dataset(key, shape=initial_shape,
                                   maxshape=max_shape, chunks=chunks, dtype=np.uint8)

        dataset = h5_file[key]
        dataset.resize((dataset.shape[0] + 1, shape[0], shape[1]))
        dataset[-1, :, :] = frame


def create_incremented_h5(store_path, base_name):
    while True:
        h5_path = os.path.join(store_path, f"{base_name}.h5")
        if not os.path.exists(h5_path):
            return h5_path
