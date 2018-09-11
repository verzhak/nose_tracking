#!/usr/bin/env python3

import sys
import cv2
import numpy as np

nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
cap = cv2.VideoCapture(0)
ds_factor = 0.5

while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx = ds_factor, fy = ds_factor, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in nose_cascade.detectMultiScale(gray, 1.3, 5):

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        break

    cv2.imshow("Nose Detector", frame)

    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()

