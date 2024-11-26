# Importing the neccesary modules
import time, math
import board, busio, adafruit_vl53l1x
import numpy as np

from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_ROTATION_VECTOR

# Creating sensor instances and setting parameters
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
bno = BNO08X_I2C(i2c)

bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)

vl53 = adafruit_vl53l1x.VL53L1X(i2c)

vl53.distance_mode = 2
vl53.timing_budget = 100
    
# Converting quaternion points from sensors to Euler points for use
def quaternion_to_euler(quaternion):
    x = quaternion[0]
    y = quaternion[1]
    z = quaternion[2]
    w = quaternion[3]
    
    sinr_comp = 2 * (w * x + y * z)
    cosr_comp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_comp, cosr_comp)
    
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)
    
    siny_comp = 2 * (w * z + x * y)
    cosy_comp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_comp, cosy_comp)

    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg

# Invoking the sensors to collect distance and orientation data
def get_data():
    try:
        distance = 0
        vl53.start_ranging()
        if vl53.data_ready:
            distance = vl53.distance
            vl53.clear_interrupt()
        roll, pitch, yaw = quaternion_to_euler(bno.quaternion)
        
        return distance, yaw, pitch, roll
    except:
        print("IO Error Occurred - Check Connections")