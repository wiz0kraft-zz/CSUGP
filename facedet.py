#!/usr/bin/env python

'''
face detection using haar cascades

USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import math

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

if __name__ == '__main__':
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn  = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)
    #Dabangg/Dabangg
    cam = cv2.VideoCapture('/home/wizkraft/Videos/MrAndMrsSmith/m.mp4')
    i=0;
    j=1
    M = [0] * 20
    index = 0; # index for array M
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        t = clock()
        if i%15 == 0:
            rects = detect(gray, cascade)
            if not type(rects) is list : 
                crop = img[rects[0,1]:rects[0,3], rects[0,0]:rects[0,2]]
                #cv2.imwrite("/home/wizkraft/Desktop/Faces/StrangerThingsS01E01/"+str(j)+".jpg",crop_img)
                height, width, channels = crop.shape
                if height<250:
                    continue
                #f = 100.0/height
                #pre = cv2.resize(crop,None,fx=f, fy=f, interpolation = cv2.INTER_AREA)
                #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                #result = cv2.equalizeHist(crop)
                cv2.imwrite("/home/wizkraft/Desktop/Faces/MrAndMrsSmith/"+str(j)+".jpg",crop)              
                j = j+1
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))

        dt = clock() - t
        if i%14400 ==0:
            M[index]=int(j)
            index = index + 1
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        cv2.imshow('facedetect', vis)
        i = i+1
        np.savetxt('/home/wizkraft/Desktop/Faces/MrAndMrsSmith/images.txt',M)


        if cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
