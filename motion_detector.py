import sys
import cv2
import numpy as np
import time
import math
import imutils

#Camera defaults to '0' (USB webcam 1) if not specified as part of call argument.
if len(sys.argv > 1):
    camera = sys.argv[1]
else:
    camera = 0

white_settle = True
framezero = None
reset = True
period = 1
min_area = (64*48) *2

cap = cv2.VideoCapture(camera)
while(cap.isOpened()):
    
    proctime = time.process_time()
    
    if reset == True:
        limit = int(proctime) + period
        reset = False
        framezero = None

    if int(proctime) == limit:
        reset = True

    _, frame = cap.read()
    frame = cv2.resize(frame, (640,480))

    if white_settle:
        time.sleep(3)
        for i in range(20):
            __,frame = cap.read()
        white_settle = False

    text = "No detections"

    gr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bl_gr_frame = cv2.GaussianBlur(gr_frame, (21,21), 0)

    if framezero is None:
         framezero = bl_gr_frame #resetting initial frame
    
    dlta_frame = cv2.absdiff(framezero, gr_frame) #absolute difference between frame_zero and gray frame
    thrs_frame = cv2.threshold(dlta_frame, 80, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thrs_frame, None, iterations = 5) #dilate thresholded frame
    contours = cv2.findContours(dil_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    try:
        contours = imutils.grab_contours(contours)
    except:
        pass

    boundRect = [None]*len(contours)

    for pts in contours:
        if cv2.contourArea(pts) < min_area:
            continue
        else:
            (x,y,w,h) = cv2.boundingRect(pts)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200,20,100), 2)
            text = "Motion detected."

    cv2.putText(frame, text, (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90,90,150), 2)
    cv2.imshow("feed", cv2.resize(frame,(640,480)))
    cv2.imshow("thresh", cv2.resize(thrs_frame,(640,480)))
    cv2.imshow("delta", cv2.resize(dlta_frame,(640,480)))
    
    if cv2.waitKey(25) and 0xFF == ord('q'): #if video
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
