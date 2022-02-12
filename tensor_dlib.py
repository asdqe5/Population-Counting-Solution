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

mydb = pymysql.connect(host = 'localhost', user = 'root', password = '1111', database = 'test_pymysql', charset = 'utf8', autocommit = True)
mycursor = mydb.cursor()
mycursor.execute("DROP TABLE People_Detection")
mycursor.execute("DROP TABLE Counting")
mycursor.execute("DROP TABLE Region_Data")
mycursor.execute("CREATE TABLE People_Detection(ID INT(30), coordinate_X FLOAT(30), coordinate_Y FLOAT(30), State VARCHAR(255), Frames_In_Region INT(30), Frames INT(30))")
mycursor.execute("ALTER TABLE People_Detection max_rows=10000000 avg_row_length=6024000000")
mycursor.execute("CREATE TABLE Counting(ID INT(30), State VARCHAR(255), OUT_ONE INT(30), IN_ONE INT(30), OUT_NEW INT(30), IN_NEW INT(30), OUT_LIB INT(30), IN_LIB INT(30), OUT_SMOKE INT(30), IN_SMOKE INT(30), Seconds INT(30))")
mycursor.execute("CREATE TABLE Region_Data(ID INT(30), Time_In_Region INT(30), From_Where VARCHAR(255))")
input_video = "time9.mp4"

vs = cv2.VideoCapture(input_video)
skip_frames = 10
det = detector.PersonDetector()

# initialize the video writer (we'll instantiate later if need be)
writer = None

# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject, maxDisappeared: frame, maxDistance: distance to centroid
ct = CentroidTracker(maxDisappeared = 30, maxDistance = 40)
trackers = []
trackableObjects = {}
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
object_state = " "
totalFrames = 0 # 프레임 수
out_from_one = 0 # 원흥관 아웃
in_to_one = 0 # 원흥관 인
out_from_new = 0 # 신공학관 아웃
in_to_new = 0 # 신공학관 인
out_from_lib = 0 #중도 및 팔정도 아웃
in_to_lib = 0 #중도 및 팔정도인
in_the_smoke = 0 #흡연구역 안
out_from_smoke = 0 #흡연구역 밖
in_region_num = 0
tuple_num = 0
smoking_people = 0
smoking_stay = []
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
        writer = cv2.VideoWriter("output_9am.avi", fourcc, 30, (W, H), True)

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
            (startY, startX, endY, endX) = box.astype("int")
            # construct a dlib rectangle object from the bounding
            # box coordinates and then start the dlib correlation
            # tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
            tracker.start_track(rgb, rect)
            # add the tracker to our list of trackers so we can
            # utilize it during skip frames
            trackers.append(tracker)
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    cv2.rectangle(frame, (0, H//2 - 140), (150, H//2 - 40), (50, 250, 255), 2) #원흥관 네모
    cv2.rectangle(frame, (230, 110), (660, 240), (0, 120, 255), 2) #흡연구역 영역
    cv2.rectangle(frame, (W - 550, 80), (W - 300, 230), (120, 0, 255), 2) #중앙도서관 네모
    cv2.rectangle(frame, (0, H - 220), (W, H - 90), (120, 120, 255), 2) #신공학관 네모
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        object_state = " "
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            x = to.centroids[-1][0]
            y = to.centroids[-1][1]
            direction_y = centroid[1] - y
            direction_x = centroid[0] - x
            to.centroids.append(centroid)

            #centroid가 영상 내에 있는지 확인
            if centroid[0] >= 0 and centroid[0] <= W and centroid[1] >= 0 and centroid[1] <= H:
                to.counted = True
            else:
                to.counted = False
            #centroid가 영상 내에 있다면
            if to.counted:
                #원흥관 IN/OUT
                if not to.counted_one:
                    if x >= 0 and x <= 150 and y <= H//2 - 40 and y >= H//2 - 140:
                        if centroid[0] >= 150 or centroid[1] >= H//2 - 40 or centroid[1] <= H//2 - 140:
                            out_from_one += 1
                            object_state = "Out_From_One"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_one = True
                    else:
                        if centroid[0] >= 0 and centroid[0] <= 150 and centroid[1] >= H//2 - 140 and centroid[1] <= H//2 - 40:
                            in_to_one += 1
                            object_state = "In_To_One"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_one = True
                #신공학관 IN/OUT
                if not to.counted_new:
                    if x >= 0 and x <= W and y >= H - 220 and y <= H - 90:
                        if centroid[0] <= 0 or centroid[1] <= H - 220:
                            out_from_new += 1
                            object_state = "Out_From_New"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_new = True
                    else:
                        if centroid[0] >= 0 and centroid[0] <= W and centroid[1] >= H - 220 and centroid[1] <= H - 90:
                            in_to_new += 1
                            object_state = "In_To_New"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_new = True
                #중도 IN/OUT
                if not to.counted_lib:
                    if x >= W - 550 and x <= W - 300 and y >= 80 and y <= 230:
                        if centroid[0] <= W - 550 or centroid[1] >= 230:
                            out_from_lib += 1
                            object_state = "Out_From_Lib"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_lib = True
                    else:
                        if centroid[0] >= W - 550 and centroid[0] <= W - 300 and centroid[1] >= 80 and centroid[1] <= 230:
                            in_to_lib += 1
                            object_state = "In_To_Lib"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_lib = True
                #흡연구역 IN/OUT
                if not to.counted_smoke:
                    if totalFrames <= 2:
                        if centroid[0] >= 230 and centroid[0] <= 660 and centroid[1] >= 110 and centroid[1] <= 240:
                            in_the_smoke += 1
                            object_state = "In_The_Smoke"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_smoke = True
                    else:
                        if centroid[0] >= 230 and centroid[0] <= 660 and centroid[1] >= 110 and centroid[1] <= 240:
                            object_state = "In_The_Smoke"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_smoke = True
                        if x <= 230 or x >= 660 or y <= 110 or y >= 240:
                            if centroid[0] >= 230 and centroid[0] <= 660 and centroid[1] >= 110 and centroid[1] <= 240:
                                in_the_smoke += 1
                                object_state = "In_The_Smoke"
                                mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                                to.counted_smoke = True
                                to.counted_lib = False
                                to.counted_new = False
                                to.counted_one = False
                        elif x >= 230 and x <= 660 and y >= 110 and y <= 240:
                            if centroid[0] <= 230 or centroid[0] >= 660 or centroid[1] <= 110 or centroid[1] >= 240:
                                out_from_smoke += 1
                                object_state = "Out_From_Smoke"
                                mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                                to.counted_smoke = False
                else:
                    if x >= 230 and x <= 660 and y >= 110 and y <= 240:
                        if centroid[0] <= 230 or centroid[0] >= 660 or centroid[1] <= 110 or centroid[1] >= 240:
                            out_from_smoke += 1
                            object_state = "Out_From_Smoke"
                            mycursor.execute("INSERT INTO Counting VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (objectID, object_state, out_from_one, in_to_one, out_from_new, in_to_new, out_from_lib, in_to_lib, out_from_smoke, in_the_smoke, totalFrames // 30))
                            to.counted_smoke = False
                ##현재 object가 region안에 있다면 frame 수를 저장하고 db에 전송한다.
                if centroid[0] >= 230 and centroid[0] <= 660 and centroid[1] >= 110 and centroid[1] <= 240:
                    to.frame_in_region += 1
                    if to.frame_in_region == 900:
                        smoking_people += 1
                        smoking_stay.append(objectID)

        mycursor.execute("INSERT INTO People_Detection VALUES (%s, %s, %s, %s, %s, %s)", (objectID, float(centroid[0]), float(centroid[1]), object_state, to.frame_in_region, totalFrames))
        trackableObjects[objectID] = to
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 2, (100, 255, 50), -1)

    info1 = [
		("Wonheungwan_OUT", out_from_one),
		("Wonheungwan_IN", in_to_one),
	]
    info2 = [
        ("Singonghaggwan_OUT", out_from_new),
        ("Singonghaggwan_IN", in_to_new),
    ]
    info3 = [
        ("Library_OUT", out_from_lib),
        ("Library_IN", in_to_lib),
    ]
    info4 = [
        ("Total_Smoking_num", smoking_people),
        ("SmokingZone_ID", smoking_stay),
    ]

    for (i, (k, v)) in enumerate(info1):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 250, 255), 2)

    for (i, (k, v)) in enumerate(info2):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (300, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 2)

    for (i, (k, v)) in enumerate(info3):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (600, ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 0, 255), 2)

    for (i, (k, v)) in enumerate(info4):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, ((i * 20) + H - 70)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 255), 2)

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

# 저장된 database를 통해서 region에 대한 분석
in_region_num = mycursor.execute("SELECT ID From People_Detection WHERE State = 'In_The_Smoke'")
in_region_data = mycursor.fetchall()
for num in range(in_region_num):
    tuple_num = mycursor.execute("SELECT * From People_Detection WHERE ID = %s", (in_region_data[num][0]))
    tuple_data = mycursor.fetchall()
    region_time = tuple_data[tuple_num - 1][4]
    region_id = tuple_data[tuple_num - 1][0]
    region_state_num = mycursor.execute("SELECT * From People_Detection WHERE State != ' ' and ID = %s", (in_region_data[num][0]))
    region_state_data = mycursor.fetchall()
    for num2 in range(region_state_num):
        if(region_state_data[num2][3] == 'Out_From_One'):
            region_state = 'Out_From_One'
            break
        elif(region_state_data[num2][3] == 'Out_From_New'):
            region_state = 'Out_From_New'
            break
        elif(region_state_data[num2][3] == 'Out_From_Lib'):
            region_state = 'Out_From_Lib'
            break
        else:
            region_state = 'Not_Found'
    mycursor.execute("INSERT INTO Region_Data Values (%s, %s, %s)", (region_id, region_time, region_state))

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
