import cv2

cap = cv2.VideoCapture("Squat.mp4")
print("Opened:", cap.isOpened())

ret, frame = cap.read()
print("Frame read:", ret)