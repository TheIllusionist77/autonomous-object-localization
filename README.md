# autonomous-object-localization
A work-in-progress autonomous object localization system that can identify and localize objects in a user's surroundings.

/images - A folder of images used for calibration, with one example undistorted image.
benchmarks.py - Runs YOLO export framework benchmarks on CPU.
camera_calibration.py - Derives the intrinsic matrix of a camera, taken from OpenCV.
image_capture.py - Takes a picture using a connected camera for calibration use.
intrinsic.npy - The intrinsic matrix of the Raspberry Pi v3 Wide camera.
main.py - Calculates position of user and objects and exports data.
object_detection.py - Uses YOLO for object inference.
odometry_visualizer.py - Parses data from main.py to draw a 2D visualization of movement and target object locations.
sensors.py - Reads and writes to the IMU and ToF sensors.
visual_odometry.py - Conducts monocular visual odometry, adapted from Nicolai Nielsen.
