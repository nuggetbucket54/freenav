import cv2
import numpy as np
import time
import pyautogui
#import pygame
import sys
from threading import Timer

'''
Notice: this program is largely dependent on the parameters set for the HSV values.
Different lighting levels can cause very different results from the program.
It's best to first calibrate the correct HSV parameters before starting.
'''

def clicker():
    global count
    global text
    if count == 0:
        pyautogui.click()
        text = "Closed"
    else:
        text = "Open"
    Timer(0.25, clicker).start()

camera = cv2.VideoCapture(0)
blur = ((2, 2), 1)
erode = (5, 5)
dilate = (3, 3)
startflag = True
text = ""

while 1:
    _, frame = camera.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #settings for brighter lighting conditions
    bottom = np.array([0, 26, 133], "uint8")
    top = np.array([179, 255, 255], "uint8")

    #settings for darker lighting conditions (irrelevant but kept for reference)
    #bottom = np.array([0, 67, 49], "uint8")
    #top = np.array([179, 255, 255], "uint8")

    #process of binarizing webcam footage
    hand = cv2.inRange(hsv, bottom, top)
    kernel = np.ones((5, 5), np.uint8)
    #binarizing by dilation/erosion experiment (did not work as well as a simpler method)
    #blurred = cv2.dilate(cv2.erode(hand, kernel, iterations = 1), kernel, iterations = 1)
    blurred = cv2.blur(hand, (30, 30))
    _, cut = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)

    #determining contour around hand based off binarized image
    contourlist, _ = cv2.findContours(cut, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contourlist:
        continue
    contourlist = max(contourlist, key=lambda x: cv2.contourArea(x))

    #creating a convexhull around contour
    hull = cv2.convexHull(contourlist, returnPoints=False)
    deviations = cv2.convexityDefects(contourlist, hull)

    #if webcam footage failed to load, deviations not found, etc.
    if deviations is not None:
        count = 0

    #spaces between fingers treated as triangles and detected if angle is within limits
    try:
        for i in range(deviations.shape[0]):
            s, e, f, d = deviations[i][0]

            #base, tip, far are base of finger, fingertip, farthest point of finger respectively
            base = tuple(contourlist[s][0])
            tip = tuple(contourlist[e][0])
            far = tuple(contourlist[f][0])


            a = ((tip[0] - base[0]) ** 2 + (tip[1] - base[1]) ** 2) ** 0.5
            b = ((far[0] - base[0]) ** 2 + (far[1] - base[1]) ** 2) ** 0.5
            c = ((tip[0] - far[0]) ** 2 + (tip[1] - far[1]) ** 2) ** 0.5
            ang = (np.cos((b ** 2 + c ** 2 -a ** 2)/(2 * b * c))) ** -1

            #convert radians to degrees
            #counts fingers by assuming nobody has 90° gap between fingers
            if ang * 180 / np.pi <= 90 and a > 100 and b > 100:
                count += 1
                cv2.circle(frame, far, 4, [0, 255, 255], -1)
            if count > 0:
                count += 1
    except:
        continue

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(frame, contourlist, -1, [255, 255, 0], -1)
    cv2.imshow("FreeNav", frame)

    count //= 5

    if startflag:
        clicker()
        startflag = False

    cv2.waitKey(1)
