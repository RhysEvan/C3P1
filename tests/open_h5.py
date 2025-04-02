import os
import h5py
import matplotlib.pyplot as plt

# Function to visualize and save the first image for each key in the H5 file
def visualize_and_save_first_images(h5_file_path, save_dir):
    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open the H5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Iterate over each key in the file
        for key in h5_file.keys():
            # Extract the image data for the current key
            image_data = h5_file[key]

            for i in range(len(image_data)):
                first_image = image_data[i]

                # Print min/max values of the first image for debugging
                print(first_image.min())
                print(first_image.max())
                print(f"Loaded image from key: {key}, shape: {image_data.shape}")

                # Visualize the first image
                plt.imshow(first_image, cmap='gray')
                plt.title(f'First image for key: {key}')
                plt.axis('off')  # Hide the axis

                # Save the image as a jpg file
                image_filename = os.path.join(save_dir, f'{key}_first_image_{i}.jpg')
                plt.savefig(image_filename, format='jpg', bbox_inches='tight', pad_inches=0)
                print(f"Saved image as {image_filename}")

                # Show the image
                # plt.show()

                # Clear the figure to avoid memory issues when processing multiple images
                # plt.clf()

# Function to visualize the first image for each key in the H5 file
def return_data(h5_file_path):
    # Open the H5 file
    with h5py.File(h5_file_path, 'r') as h5_file:
        # Iterate over each key in the file
        calibration = h5_file["camera_calibration/camera_parameters"]
        print(calibration.keys())
        c = calibration["c"]
        print(c[:])
        f = calibration["f"]
        print(f[:])
        rad_dst = calibration["radial_dist_coeffs"]
        print(rad_dst[:])
        tan_dst = calibration["tangential_dist_coeffs"]
        print(tan_dst[:])
        cam_in = [
            f[0], 0, c[0],
            0, f[1], c[1],
            0, 0, 1
        ]
        cam_dist = [rad_dst[0], rad_dst[1], tan_dst[0], tan_dst[1], rad_dst[2]]
        print("c_std")
        print(calibration["c_std"][:])
        print("f_std")
        print(calibration["f_std"][:])



# Example usage
h5_file_path = r'C:\Users\InViLab\Desktop\3-View_Application\CPC_CamProCam_UAntwerp\static\1_calibration_data\turntable\scan.h5'  # Replace with your H5 file path
save_path = r'C:\Users\InViLab\Desktop\3-View_Application\CPC_CamProCam_UAntwerp\static\1_calibration_data\turntable/'
visualize_and_save_first_images(h5_file_path, save_path)
# return_data(h5_file_path)
