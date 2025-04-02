import numpy as np

# Assuming 'data.npy' is your numpy file
file_path = r'C:\Users\Rhys\Documents\GitHub\CPC_CamProCam_UAntwerp\temp\github_data_example\proj_dist.npy'

# Load data from the file
loaded_data = np.load(file_path)

# Print the loaded data
print("Loaded data:")
print(loaded_data)