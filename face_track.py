import numpy as np
import cv2 as cv
import sys

cv.namedWindow("tracking")
camera = cv.VideoCapture(0)
tracker = cv.MultiTracker_create()
init_once = False

ok, image=camera.read()
if not ok:
    print('Failed to read video')
    exit()

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

frame_count=0
while camera.isOpened():
    ok, image=camera.read()
    if not ok:
        print ('no image to read')
        break

    if not init_once:
        tracker = cv.MultiTracker_create()
        faces = face_cascade.detectMultiScale(image)
        for (x,y,w,h) in faces:
            ok = tracker.add(cv.TrackerKCF_create(), image, (x,y,w,h))
        init_once = True

    ok, boxes = tracker.update(image)
    print (ok, boxes)

    count=0
    for newbox in boxes:
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv.rectangle(image, p1, p2, (200,count*100,count*100),3)
        count+=1

    frame_count+=1
    if frame_count==25:
        frame_count=0
        init_once=False
    cv.imshow('tracking', image)
    k = cv.waitKey(1)
    if k == 27 : break # esc pressed
