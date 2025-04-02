import os
import glob

def rename_files(directory):
    # Change the working directory to the target folder
    os.chdir(directory)

    # Use glob to find all files in the directory
    all_files = sorted(glob.glob('*'))

    # Extract the img_white and img_black (assuming these are the last two files)
    if len(all_files) < 2:
        print("Not enough files in the directory.")
        return

    img_white = all_files[-2]
    img_black = all_files[-1]

    # Rename img_white and img_black
    os.rename(img_white, 'L_pattern_white.tiff')
    os.rename(img_black, 'L_pattern_black.tiff')

    # Process the remaining files
    gray_images = all_files[:-2]

    # Assuming gray_images are all of the same length and the correct files to be renamed
    if not gray_images:
        print("No gray images found in the directory.")
        return

    gray_images_ver = gray_images[0:22]
    gray_images_hor = gray_images[22:]

    gray_images_hor_inv = gray_images_hor[1::2]
    gray_images_ver_inv = gray_images_ver[1::2]
    gray_images_hor = gray_images_hor[::2]
    gray_images_ver = gray_images_ver[::2]

    # Rename gray_images_hor
    for i, img in enumerate(gray_images_hor):
        new_name = f'L_pattern_H_{i:02d}.tiff'
        os.rename(img, new_name)

    # Rename gray_images_ver
    for i, img in enumerate(gray_images_ver):
        new_name = f'L_pattern_V_{i:02d}.tiff'
        os.rename(img, new_name)

    # Rename gray_images_hor_inv
    for i, img in enumerate(gray_images_hor_inv):
        new_name = f'L_pattern_H_I_{i:02d}.tiff'
        os.rename(img, new_name)

    # Rename gray_images_ver_inv
    for i, img in enumerate(gray_images_ver_inv):
        new_name = f'L_pattern_V_I_{i:02d}.tiff'
        os.rename(img, new_name)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder: ")
    rename_files(folder_path)
