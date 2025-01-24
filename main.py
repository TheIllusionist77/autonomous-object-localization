# Importing the neccesary modules
import cv2, os, datetime
import numpy as np

from picamera2 import Picamera2
from visual_odometry import *
from object_detection import *
from sensors import *
from adafruit_vl53l1x import *

# Storing the intrinsic matrix of the camera
with open("intrinsic.npy", "rb") as f:
    intrinsic = np.load(f)

# Defining some variables and lists
skip_frames = 2
data_dir = ""
vo = CameraPoses(data_dir, skip_frames, intrinsic)

gt_path = []
estimated_path = []
camera_pose_list = []
start_pose = np.ones((3, 4))
start_translation = np.zeros((3, 1))
start_rotation = np.identity(3)
start_pose = np.concatenate((start_rotation, start_translation), axis = 1)

RED = (255, 0, 0)
CAMERA_FOV = 107
SCREEN_SIZE = 640
SAVE_FILE = "odometry_logs/log" + str(datetime.datetime.now().strftime("%m%d%y-%H%M%S"))

# Initializing the camera stream
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main = {"format": "RGB888", "size": (SCREEN_SIZE, SCREEN_SIZE)}))
picam2.start()

# Allows us to convert degrees to region of interest area and vice versa
def conversion(degrees = None, area = None):
    if degrees != None:
        return 1.25 ** (degrees - 2)
    elif area != None:
        return (math.log(area) / math.log(1.25)) + 2
    
# Returns the ratio of the camera view that the ToF ROI covers and vice versa
def distance_fov(degrees = None, area = None):
    if degrees != None:
        return (degrees ** 2) / (CAMERA_FOV ** 2)
    if area != None:
        return CAMERA_FOV * math.sqrt(area)

# Returns if any one of two given rects encompass the other
def check_containment(delta_x1, delta_y1, delta_x2, delta_y2, center_x1, center_y1, center_x2, center_y2):
    rect1_corners = [
        (delta_x1, delta_y1),  # Top-left
        (delta_x2, delta_y1),  # Top-right
        (delta_x2, delta_y2),  # Bottom-right
        (delta_x1, delta_y2)   # Bottom-left
    ]

    rect2_corners = [
        (center_x1, center_y1),  # Top-left
        (center_x2, center_y1),  # Top-right
        (center_x2, center_y2),  # Bottom-right
        (center_x1, center_y2)   # Bottom-left
    ]

    def is_point_in_rectangle(px, py, x1, y1, x2, y2):
        return x1 <= px <= x2 and y1 <= py <= y2

    def is_rectangle_in_rectangle(rect, x1, y1, x2, y2):
        return all(is_point_in_rectangle(x, y, x1, y1, x2, y2) for x, y in rect)

    # Check if rect1 is inside rect2 or rect2 is inside rect1
    return is_rectangle_in_rectangle(rect1_corners, center_x1, center_y1, center_x2, center_y2) or is_rectangle_in_rectangle(rect2_corners, delta_x1, delta_y1, delta_x2, delta_y2)

# Optimizes the ToF ROI for a given bounding box
def calculate_tof_region(x1, y1, x2, y2):
    # Calculate the 16x16 ToF FoV in degrees and pixels
    max_grid_size = 16
    max_area = max_grid_size ** 2
    max_degrees = conversion(area=max_area)
    max_fov_ratio = distance_fov(max_degrees)
    max_fov_pixel_width = max_fov_ratio * SCREEN_SIZE
    max_fov_pixel_height = max_fov_ratio * SCREEN_SIZE

    # Find the portion of the bounding box within the ToF 16x16 field of view
    tof_fov_x1 = SCREEN_SIZE / 2 - max_fov_pixel_width / 2
    tof_fov_y1 = SCREEN_SIZE / 2 - max_fov_pixel_height / 2
    tof_fov_x2 = SCREEN_SIZE / 2 + max_fov_pixel_width / 2
    tof_fov_y2 = SCREEN_SIZE / 2 + max_fov_pixel_height / 2

    overlap_x1 = max(x1, tof_fov_x1)
    overlap_y1 = max(y1, tof_fov_y1)
    overlap_x2 = min(x2, tof_fov_x2)
    overlap_y2 = min(y2, tof_fov_y2)

    # Check if there is an overlap
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        return True

    overlap_width = overlap_x2 - overlap_x1
    overlap_height = overlap_y2 - overlap_y1

    # Try fitting all possible grid sizes (16x16 to 4x4)
    for grid_size in range(16, 3, -1):  # Decrement from 16x16 to 4x4
        area = grid_size ** 2
        degrees = conversion(area=area)
        fov_ratio = distance_fov(degrees)
        fov_pixel_width = fov_ratio * SCREEN_SIZE
        fov_pixel_height = fov_ratio * SCREEN_SIZE

        # Check if this grid size fits within the overlap box
        if fov_pixel_width <= overlap_width and fov_pixel_height <= overlap_height:
            # Align the ToF region to the overlap box
            x_tof_center = (overlap_x1 + overlap_x2) / 2
            y_tof_center = (overlap_y1 + overlap_y2) / 2

            # Convert ToF center in pixels to 16x16 grid
            screen_area = overlap_width * overlap_height
            grid_degrees = distance_fov(area = (screen_area / (SCREEN_SIZE * SCREEN_SIZE)))
            grid_area = conversion(degrees = grid_degrees)
            grid_x = round(math.sqrt(grid_area))
            grid_y = round(math.sqrt(grid_area))

            # Convert the center grid to ToF ROI index
            roi_center_index = roi_index[(grid_y + round((grid_size - 1) / 2 - 1)) * 16 + round(grid_x + (grid_size - 1) / 2)]

            # Return the final settings
            return grid_size, roi_center_index
    
    return False

# Defining some more variables
process_frames = False
old_frame = None
new_frame = None
frame_counter = 0
running = True

cur_pose = start_pose
x, y, disp_x, disp_y = 0, 0, 0, 0
distance, yaw, last_x, last_y = 0, 0, 0, 0

delta_x1 = 0
delta_y1 = 0
delta_x2 = 0
delta_y2 = 0

center_x1 = 0
center_y1 = 0
center_x2 = 0
center_y2 = 0

roi_index = {
    1: 128,   2: 136,   3: 144,   4: 152,   5: 160,   6: 168,   7: 176,   8: 184,    9: 192,  10: 200,  11: 208,  12: 216,  13: 224,  14: 232,  15: 240,  16: 248,
   17: 129,  18: 137,  19: 145,  20: 153,  21: 161,  22: 169,  23: 177,  24: 185,   25: 193,  26: 201,  27: 209,  28: 217,  29: 225,  30: 233,  31: 241,  32: 249,
   33: 130,  34: 138,  35: 146,  36: 154,  37: 162,  38: 170,  39: 178,  40: 186,   41: 194,  42: 202,  43: 210,  44: 218,  45: 226,  46: 234,  47: 242,  48: 250,
   49: 131,  50: 139,  51: 147,  52: 155,  53: 163,  54: 171,  55: 179,  56: 187,   57: 195,  58: 203,  59: 211,  60: 219,  61: 227,  62: 235,  63: 243,  64: 251,
   65: 132,  66: 140,  67: 148,  68: 156,  69: 164,  70: 172,  71: 180,  72: 188,   73: 196,  74: 204,  75: 212,  76: 220,  77: 228,  78: 236,  79: 244,  80: 252,
   81: 133,  82: 141,  83: 149,  84: 157,  85: 165,  86: 173,  87: 181,  88: 189,   89: 197,  90: 205,  91: 213,  92: 221,  93: 229,  94: 237,  95: 245,  96: 253,
   97: 134,  98: 142,  99: 150, 100: 158, 101: 166, 102: 174, 103: 182, 104: 190,  105: 198, 106: 206, 107: 214, 108: 222, 109: 230, 110: 238, 111: 246, 112: 254,
  113: 135, 114: 143, 115: 151, 116: 159, 117: 167, 118: 175, 119: 183, 120: 191,  121: 199, 122: 207, 123: 215, 124: 223, 125: 231, 126: 239, 127: 247, 128: 255,
  129: 127, 130: 119, 131: 111, 132: 103, 133:  95, 134:  87, 135:  79, 136:  71,  137:  63, 138:  55, 139:  47, 140:  39, 141:  31, 142:  23, 143:  15, 144:   7,
  145: 126, 146: 118, 147: 110, 148: 102, 149:  94, 150:  86, 151:  78, 152:  70,  153:  62, 154:  54, 155:  46, 156:  38, 157:  30, 158:  22, 159:  14, 160:   6,
  161: 125, 162: 117, 163: 109, 164: 101, 165:  93, 166:  85, 167:  77, 168:  69,  169:  61, 170:  53, 171:  45, 172:  37, 173:  29, 174:  21, 175:  13, 176:   5,
  177: 124, 178: 116, 179: 108, 180: 100, 181:  92, 182:  84, 183:  76, 184:  68,  185:  60, 186:  52, 187:  44, 188:  36, 189:  28, 190:  20, 191:  12, 192:   4,
  193: 123, 194: 115, 195: 107, 196:  99, 197:  91, 198:  83, 199:  75, 200:  67,  201:  59, 202:  51, 203:  43, 204:  35, 205:  27, 206:  19, 207:  11, 208:   3,
  209: 122, 210: 114, 211: 106, 212:  98, 213:  90, 214:  82, 215:  74, 216:  66,  217:  58, 218:  50, 219:  42, 220:  34, 221:  26, 222:  18, 223:  10, 224:   2,
  225: 121, 226: 113, 227: 105, 228:  97, 229:  89, 230:  81, 231:  73, 232:  65,  233:  57, 234:  49, 235:  41, 236:  33, 237:  25, 238:  17, 239:   9, 240:   1,
  241: 120, 242: 112, 243: 104, 244:  96, 245:  88, 246:  80, 247:  72, 248:  64,  249:  56, 250:  48, 251:  40, 252:  32, 253:  24, 254:  16, 255:   8, 256:   0
}

roi_xy = (16, 16)
roi_center = 199

raw_x, raw_y, raw_z = 0, 0, 0
  
# Main system loop
while running:
    # Capturing an image from the camera
    new_frame = picam2.capture_array()
    new_frame = new_frame[0:SCREEN_SIZE, 0:SCREEN_SIZE]
    frame_counter += 1
    start = time.perf_counter()
    
    # Extrapolation of displacement from two images, credit to Nicolai Nielsen
    if process_frames:
        q1, q2 = vo.get_matches(old_frame, new_frame)
        if q1 is not None:
            if len(q1) > 8 and len(q2) > 8:
                transformation_matrix = vo.get_pose(q1, q2)
                if abs(transformation_matrix[0][3]) <= 10 ** 1000:
                    cur_pose = cur_pose @ transformation_matrix
        
        hom_array = np.array([[0, 0, 0, 1]])
        hom_camera_pose = np.concatenate((cur_pose, hom_array), axis = 0)
        camera_pose_list.append(hom_camera_pose)
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
        raw_x, raw_y = cur_pose[0, 3], cur_pose[2, 3]
        
        disp_x = abs(raw_x) - last_x
        last_x += disp_x
        disp_y = abs(raw_y) - last_y
        last_y += disp_y
        
    elif process_frames and ret is False:
        break
    
    # Using YOLO to find objects within the image
    annotated_frame, objects = detect_objects(new_frame, 320)
    
    # Invoking the sensors to return distance and yaw
    sensor_data = get_data()
    if sensor_data != None:
        if None not in sensor_data:
            distance, yaw, _, _ = sensor_data
            
    # Calculating true (x, y) coordinates
    x += disp_x * math.cos(math.radians(yaw)) + disp_y * math.sin(math.radians(yaw))
    y += disp_x * math.sin(math.radians(yaw)) + disp_y * math.cos(math.radians(yaw))
    
    print(round(disp_x), round(disp_y))
            
    # Checking if there are any objects within the original ToF ROI
    for item in objects:
        delta_x1 = item["BBox"][0]
        delta_y1 = item["BBox"][1]
        delta_x2 = item["BBox"][2]
        delta_y2 = item["BBox"][3]
        
        if check_containment(delta_x1, delta_y1, delta_x2, delta_y2, center_x1, center_y1, center_x2, center_y2):
            print(item["Class"])
            
        break
    
    new_roi = calculate_tof_region(delta_x1, delta_y1, delta_x2, delta_y2)
    
    if new_roi != True and new_roi != False:
        roi_xy = (new_roi[0], new_roi[0])
        roi_center = new_roi[1]
    
    # Computing the ToF ROI and writing it to the camera stream
    ratio = distance_fov(degrees = conversion(area = roi_xy[0] * roi_xy[1]))
    
    tof_fov_rad = math.radians(CAMERA_FOV)
    cam_fov_rad = math.radians(conversion(area = roi_xy[0] * roi_xy[1]))

    angular_offset_rad = math.atan(2.0 / (max(distance, 1)))

    camera_center_offset_rad = angular_offset_rad
    cam_pixels_per_radian = SCREEN_SIZE / cam_fov_rad
    pixel_offset = int(camera_center_offset_rad * cam_pixels_per_radian)

    center_x1 = int((SCREEN_SIZE / 2) - (ratio * SCREEN_SIZE / 2))
    center_y1 = int((SCREEN_SIZE / 2) - (ratio * SCREEN_SIZE / 2)) - pixel_offset
    center_x2 = int((SCREEN_SIZE / 2) + (ratio * SCREEN_SIZE / 2))
    center_y2 = int((SCREEN_SIZE / 2) + (ratio * SCREEN_SIZE / 2)) - pixel_offset
    
    cv2.rectangle(annotated_frame, (center_x1, center_y1), (center_x2, center_y2), RED, 1)
    
    old_frame = new_frame
    process_frames = True
    end = time.perf_counter()
    
    total_time = end - start
    fps = 1 / total_time
    
    # Writing text to the camera stream
    cv2.putText(annotated_frame, f"FPS: {round(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"X: {round(x)}", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(annotated_frame, f"Y: {round(y)}", (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(annotated_frame, f"D: {round(distance)}", (500, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(annotated_frame, f"A: {round(yaw)}", (500, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    cv2.imshow("Localization and Object Detection", annotated_frame)
    cv2.waitKey(5)
    
# Cleaning up OpenCV
cv2.destroyAllWindows()
