from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # 
    gray = cv.equalizeHist(gray) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        center = (x + w//2, y + h//2) #
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        eyes = eyes_cascade.detectMultiScale(gray[y:y+h, x:x+w])
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade, eyes_cascade = cv.CascadeClassifier(), cv.CascadeClassifier()
if not all(cascade.load(cv.samples.findFile(args.face_cascade if i == 0 else args.eyes_cascade)) for i, cascade in enumerate((face_cascade, eyes_cascade))):
    print('--(!)Error loading cascade')
    exit(0)

cap = cv.VideoCapture(args.camera)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break