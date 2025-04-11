import os
import cv2
import h5py
import glob
import numpy as np

def show_camera_feed(camera, dimensions):
    window_name = 'Camera Feed'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    concatenated_frame = []
    while True:
        frames = camera.GetFrame()
        for  i, frame in enumerate(frames):
            if frame is None:
                print("Frame not captured correctly.")
                break
            frame = np.array(frame)
            frame = frame.astype(np.uint8)

            # Convert to BGR for display
            bgr_frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2BGR)
            bgr_frame[frame[:,:,0] <= 5] = [255, 0, 0]
            bgr_frame[frame[:,:,0] >= 250] = [0, 0, 255]


            # Resize with known width and unknown height
            # Calculate the aspect ratio
            aspect_ratio = bgr_frame.shape[1] / bgr_frame.shape[0]
            # Calculate the new height based on the desired width
            if i == 0:
                height = int(dimensions[0] / aspect_ratio)
            bgr_frame = cv2.resize(bgr_frame, (dimensions[0], height))

            if i == 0:
                concatenated_frame = bgr_frame
            else:
                concatenated_frame =np.hstack((concatenated_frame, bgr_frame))


        # Resize to fit the specified dimensions
        bgr_frame = cv2.resize(bgr_frame, dimensions, interpolation=cv2.INTER_AREA)

        # Show in OpenCV window
        cv2.imshow(window_name, concatenated_frame)

        if cv2.waitKey(10) & 0xFF == 13:  # Exit on Enter key
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break



def extrinsic_calibration_capture(store_path,counter, cameras, identifier='black',numberofimages=20):
    if len(cameras) == 0:
        print('Error: No cameras were provided.')
        return
    base_name = "scan"
    base_name = f"{base_name}_{counter}"
    h5_path = os.path.join(store_path, f"{base_name}.h5")

    show_camera_feed(cameras, (640, 480))


    for i, camera in enumerate(cameras):
        try:
            with h5py.File(h5_path, 'a') as h5_file:
               # print(f"Capturing image")
                if i == 0:
                    key = "L_pattern_"+identifier
                elif i == 1:
                    key = "R_pattern_"+identifier
                else:
                    key = str(i)+"_pattern_"+identifier
                frame = camera.GetFrame()
                if frame is None:
                    print(f"Error: Frame not captured.")
                    continue

                frame = (frame).astype(np.uint8)

                if key not in h5_file:
                    sha = frame.shape
                    initial_shape = (0, frame.shape[0], frame.shape[1])
                    max_shape = (None, frame.shape[0], frame.shape[1])
                    chunks = (1, frame.shape[0], frame.shape[1])
                    h5_file.create_dataset(key, shape=initial_shape,
                                           maxshape=max_shape, chunks=chunks, dtype=np.uint8)

                dataset = h5_file[key]
                dataset.resize((dataset.shape[0] + 1, frame.shape[0], frame.shape[1]))
                dataset[-1, :, :] = frame
        except Exception as e:
            print(f"Error accessing HDF5 file: {e}")
            cv2.destroyAllWindows()

            return
        cv2.destroyAllWindows()



def intrinsic_calibration_capture(store_path, cameras, identifier=None,numberofimages=20):
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
                for j in range(numberofimages):
                    print(f"Capturing image {j + 1}/{numberofimages} from camera {i + 1}")
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
                        sha = frame.shape
                        initial_shape = (0, frame.shape[0], frame.shape[1])
                        max_shape = (None, frame.shape[0], frame.shape[1])
                        chunks = (1, frame.shape[0], frame.shape[1])
                        h5_file.create_dataset(key, shape=initial_shape,
                                               maxshape=max_shape, chunks=chunks, dtype=np.uint8)

                    dataset = h5_file[key]
                    dataset.resize((dataset.shape[0] + 1, frame.shape[0], frame.shape[1]))
                    dataset[-1, :, :] = frame
        except Exception as e:
            print(f"Error accessing HDF5 file: {e}")
            cv2.destroyAllWindows()

            return
        cv2.destroyAllWindows()


def create_incremented_h5(store_path, base_name):
    while True:
        h5_path = os.path.join(store_path, f"{base_name}.h5")
        if os.path.exists(h5_path):
            # give an error the file aready exists
            raise FileExistsError(f"Error: File {h5_path} already exists. remove it ore use a different path")
            # trow error

        if not os.path.exists(h5_path):
            return h5_path