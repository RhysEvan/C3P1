import cv2
import os

def capture_frame():
    # Create the directory if it doesn't exist
    save_dir = "lumix_test"
    os.makedirs(save_dir, exist_ok=True)

    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 for the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a frame
    ret, frame = cap.read()
    if ret:
        file_path = os.path.join(save_dir, "captured_frame.png")
        cv2.imwrite(file_path, frame)
        print(f"Frame saved at {file_path}")
    else:
        print("Error: Could not capture frame.")

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frame()
