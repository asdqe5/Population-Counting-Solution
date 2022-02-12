import numpy as np 
import matplotlib.pyplot as plt 
import glob
import time
import cv2
import pymysql
import dlib

import helpers
import detector
import tracker
from imutils.video import VideoStream
from imutils.video import FPS
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject

input_video = "test_new1.mp4"
vs = cv2.VideoCapture(input_video)
skip_frames = 1
det = detector.PersonDetector()

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared = 15, maxDistance = 50)
trackers = []
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown1 = 0
totalUp1 = 0
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the next frame and handle if we are reading from either
    # VideoCapture or VideoStream
    ret, frame = vs.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
    	(H, W) = frame.shape[:2]

	# if we are supposed to be writing a video to disk, initialize the writer
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output.avi", fourcc, 30, (W, H), True)
    
    # initialize the current status along with our list of bounding
	# box rectangles returned by either (1) our object detector or
	# (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
	# object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []
        detections = det.get_localization(frame)
        for i in range(len(detections)):
            box = detections[i]
            box = np.expand_dims(box, axis = 0).T
            x = np.array([[box[0], 0, box[1], 0, box[2], 0, box[3], 0]]).T
            trk = tracker.Tracker()
            trk.x_state = x
            trk.predict_only()
            #(startY, startX, endY, endX) = box.astype("int") #0 1 2 3
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            

            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(trk)
    
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    # loop over the trackers
        for trk in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"
            xx = trk.x_state
            xx = xx.T[0].tolist()
            trk.box = xx
            rects.append((xx[2], xx[0], xx[6], xx[4]))
            # update the tracker and grab the updated position
            

            # unpack the position object
            

            # add the bounding box coordinates to the rectangles list
            #rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.line(frame, (0, H//2), (W, H//2), (0, 0, 255), 4)
    #cv2.line(frame, (0, H - 2), (W, H - 2), (0, 255, 255), 4)

    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and centroid[1] < H // 2 and np.mean(y) > H // 2:
                    totalUp1 += 1
                    object_state = "Up"
                    to.counted = True
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2 and np.mean(y) < H // 2:
                    totalDown1 += 1
                    object_state = "Down"
                    to.counted = True
        
        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    
    info1 = [
		("Up1", totalUp1),
		("Down1", totalDown1),
		("Status", status),
	]

    for (i, (k, v)) in enumerate(info1):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # check to see if we should write the frame to disk
    if writer is not None:
        writer.write(frame)

	# show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break

	# increment the total number of frames processed thus far and
	# then update the FPS counter
    totalFrames += 1
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# # if we are not using a video file, stop the camera video stream
# if not args.get("input", False):
# 	vs.stop()

# otherwise, release the video file pointer
else:
	vs.release()

# close any open windows
cv2.destroyAllWindows()