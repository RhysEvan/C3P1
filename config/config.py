import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

#load the config file
# Exposure
exposure_default= config['settings']['exposure']['default']
exposure_intrinsic_default = config['settings']['exposure']['intrinsic_default']
exposure_intrinsic = config['settings']['exposure']['intrinsic']
exposure_procam = config['settings']['exposure']['procam']

# Camera
cam1_id = config['settings']['cam']['id'][0]
cam2_id = config['settings']['cam']['id'][1]
com_port = config['settings']['cam']['com_port']
com_port_default = config['settings']['cam']['com_port_default']
gain = config['settings']['cam']['gain']
cam_fixed_width = config['settings']['cam']['fixed_width']
cam_default_width = config['settings']['cam']['dim']['width']
cam_default_height = config['settings']['cam']['dim']['height']
cam_interface = config['settings']['cam']['interface']

# Calibration
cal_image_count_intrinsic = config['settings']['cal']['image_count_intrinsic']
cal_image_count_stereo_cal = config['settings']['cal']['image_count_stereo_calib']
steps_cal = config['settings']['cal']['steps_cal']
steps_scan = config['settings']['cal']['steps_scan']

# Paths
patterns_folder = config['settings']['path']['patterns_folder']
capture_folder = config['settings']['path']['capture_folder']
object_path = config['settings']['path']['object_path']

