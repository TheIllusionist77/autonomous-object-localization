import cv2
from picamera2 import Picamera2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main = {"format": "XRGB8888", "size": (640, 480)}))
picam2.start() 

count = 0

while True:
    img = picam2.capture_array()
    k = cv2.waitKey(5)

    if k == ord("s"):
        cv2.imwrite("image" + str(count) + ".jpg", img)
        print("Saved image.")
        count += 1

    cv2.imshow("Camera", img)