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
        if frame.ndim == 2:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            bgr_frame[frame <= 5] = [255, 0, 0]
            bgr_frame[frame >= 250] = [0, 0, 255]
        elif frame.ndim == 3:
            bgr_frame = frame
        else:
            raise ValueError("Unsupported frame format")

        # Resize to fit the specified dimensions
        bgr_frame = cv2.resize(bgr_frame, dimensions, interpolation=cv2.INTER_AREA)

        # Show in OpenCV window
        cv2.imshow(window_name, bgr_frame)

        if cv2.waitKey(10) & 0xFF == 13:  # Exit on Enter key
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

def intrinsic_calibration_capture(store_path, cameras, identifier=None):
    if len(cameras) == 0:
        print('Error: No cameras were provided.')
        return

    for i, camera in enumerate(cameras):
        if identifier is not None and len(identifier) == len(cameras):
            h5_path = create_incremented_h5(store_path, f"{identifier[i]}_scan")
        else:
            h5_path = create_incremented_h5(store_path, "scan")

        with h5py.File(h5_path, 'a') as h5_file:
            for j in range(15):
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
                    if frame.ndim == 2:
                        initial_shape = (0, frame.shape[0], frame.shape[1])
                        max_shape = (None, frame.shape[0], frame.shape[1])
                        chunks = (1, frame.shape[0], frame.shape[1])
                    elif frame.ndim == 3:
                        initial_shape = (0, frame.shape[0], frame.shape[1], frame.shape[2])
                        max_shape = (None, frame.shape[0], frame.shape[1], frame.shape[2])
                        chunks = (1, frame.shape[0], frame.shape[1], frame.shape[2])
                    h5_file.create_dataset(key, shape=initial_shape,
                                            maxshape=max_shape, chunks=chunks, dtype=np.uint8)

                dataset = h5_file[key]
                if frame.ndim == 2:
                    dataset.resize((dataset.shape[0] + 1, frame.shape[0], frame.shape[1]))
                elif frame.ndim == 3:
                    dataset.resize((dataset.shape[0] + 1, frame.shape[0], frame.shape[1], frame.shape[2]))
                dataset[-1, :, :] = frame

def create_incremented_h5(store_path, base_name):
    while True:
        h5_path = os.path.join(store_path, f"{base_name}.h5")
        if not os.path.exists(h5_path):
            return h5_path
