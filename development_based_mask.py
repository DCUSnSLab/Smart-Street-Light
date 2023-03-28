"""
├── yolo
│   ├── labels.txt
│   ├── yolov4-tiny.cfg
│   ├── yolov4-tiny.weights
├── people.jpg
├── people_out.jpg
├── street.jpg
├── street_out.jpg
├── video.mp4
├── video_out.avi
├── yolo_image.py
└── yolo_video.py
if program cant find yolo folder in main folder it will crash."""
# example usage: python yolo_video.py -i video.mp4 -o video_out.avi
# if you want to resize video, uncomment line 110
import argparse
import glob
import time, schedule, csv
import math
import cv2
import imutils
import numpy as np
import argparse, imutils
import time, dlib, datetime

from tracker_lib.centroidtracker import CentroidTracker
from tracker_lib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from tracker_lib.mailer import Mailer
from tracker_lib import config, thread
from itertools import zip_longest

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def detect():
    '''
    Video Directory
    or
    RTSP IP
    '''
    #input_video_dir = "./videos/example_01.mp4"
    input_video_dir = "./videos/test_2.mp4"
    #input_video_dir = "./videos/MOT20-02-raw.webm"
    #input_video_dir = "./videos/055_20210615_120644.avi"
    # input_video_dir = "rtsp://admin:SnSlab20121208!@192.168.0.79:554/profile2/media.smp"
    output_video_dir = "./result/output.mp4"

    '''
    YOLOv4 Parameter
    '''
    CONFIDENCE_THRESHOLD = 0.7
    NMS_THRESHOLD = 0.4
    enable_display = 1

    '''
    If you wanna using camera change the annotation
    '''
    vc = cv2.VideoCapture(input_video_dir)
    # vc = cv2.VideoCapture(0)

    '''
    Neural Networks Directory
    '''
    mask_weights = glob.glob("yolo_mask/*.weights")[0]
    mask_labels = glob.glob("yolo_mask/*.txt")[0]
    mask_cfg = glob.glob("yolo_mask/*.cfg")[0]

    yolo_weights = glob.glob("yolo/yolov4.weights")[0]
    yolo_labels = glob.glob("yolo/labels.txt")[0]
    yolo_cfg = glob.glob("yolo/yolov4.cfg")[0]

    age_proto = "caffe/age_deploy.prototxt"
    age_model = "caffe/age_net.caffemodel"

    gender_proto = "caffe/gender_deploy.prototxt"
    gender_model = "caffe/gender_net.caffemodel"


    print("You are now using Mask Predict File\n\t{} mask_weights\n\t{} mask_cfg\n\t{} mask_labels.".format(mask_weights,
                                                                                                          mask_cfg,
                                                                                                          mask_labels))
    print("You are now using YOLO Predict File\n\t{} yolo_weights\n\t{} yolo_cfg\n\t{} yolo_labels.".format(yolo_weights,
                                                                                                          yolo_cfg,
                                                                                                          yolo_labels))

    mask_class_names = list()
    yolo_class_names = list()
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    with open(mask_labels, "r") as f:
        mask_class_names = [cname.strip() for cname in f.readlines()]
    with open(yolo_labels, "r") as f:
        yolo_class_names = [cname.strip() for cname in f.readlines()]

    Mask_Box_COLORS = np.random.randint(0, 255, size=(len(mask_class_names), 3), dtype="uint8")
    Yolo_Box_COLORS = np.random.randint(0, 255, size=(len(yolo_class_names), 3), dtype="uint8")

    '''
    Create Mask Predict Network
    '''
    mask_net = cv2.dnn.readNetFromDarknet(mask_cfg, mask_weights)
    mask_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    mask_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    mask_layer = mask_net.getLayerNames()
    mask_layer = [mask_layer[i[0] - 1] for i in mask_net.getUnconnectedOutLayers()]

    '''
    Create YOLO Predict Network
    '''
    yolo_net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    yolo_layer = yolo_net.getLayerNames()
    yolo_layer = [yolo_layer[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    '''
    Create Age & Gender Predict Network
    '''
    ageNet = cv2.dnn.readNet(age_model, age_proto)
    genderNet = cv2.dnn.readNet(gender_model, gender_proto)

    writer = None
    person_count = 0

    '''
    Tracker
    '''
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    totalFrames = 0
    totalDown = 0
    totalUp = 0
    x = []
    empty = []
    empty1 = []

    fps = FPS().start()

    if config.Thread:
        vs = thread.ThreadingClass(config.url)

    while True:
        ret, frm = vc.read()

        #frm = imutils.resize(frm, width=720)
        frm = cv2.resize(frm, (1280, 720))
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

        (H, W) = frm.shape[:2]

        rects = []
        status = "Waiting"
        blob = cv2.dnn.blobFromImage(frm, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        mask_net.setInput(blob)
        yolo_net.setInput(blob)

        start_time = time.time()
        Mask_layerOutputs = mask_net.forward(mask_layer)
        Yolo_layerOutputs = yolo_net.forward(yolo_layer)
        end_time = time.time()

        mask_boxes = []
        mask_classIds = []
        mask_confidences = []

        yolo_boxes = []
        yolo_classIds = []
        yolo_confidences = []
        print("Waiting")

        for output in Mask_layerOutputs:
            for mask_detection in output:
                status = "Detecting"

                mask_scores = mask_detection[5:]
                mask_classID = np.argmax(mask_scores)
                mask_confidence = mask_scores[mask_classID]

                if mask_confidence > CONFIDENCE_THRESHOLD:
                    mask_box = mask_detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = mask_box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    mask_boxes.append([x, y, int(width), int(height)])
                    mask_classIds.append(mask_classID)
                    mask_confidences.append(float(mask_confidence))

        for output in Yolo_layerOutputs:
            for yolo_detection in output:
                status = "Detecting"
                yolo_scores = yolo_detection[5:]
                yolo_classID = np.argmax(yolo_scores)
                yolo_confidence = yolo_scores[yolo_classID]

                if yolo_confidence > CONFIDENCE_THRESHOLD and yolo_classID == 0:

                    yolo_box = yolo_detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = yolo_box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    yolo_boxes.append([x, y, int(width), int(height)])
                    yolo_classIds.append(yolo_classID)
                    yolo_confidences.append(float(yolo_confidence))

        mask_idxs = cv2.dnn.NMSBoxes(mask_boxes, mask_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        yolo_idxs = cv2.dnn.NMSBoxes(yolo_boxes, yolo_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        if len(mask_idxs) > 0:
            for i in mask_idxs.flatten():
                (x, y) = (mask_boxes[i][0], mask_boxes[i][1])
                (w, h) = (mask_boxes[i][2], mask_boxes[i][3])

                color = [int(c) for c in Mask_Box_COLORS[mask_classIds[i]]]
                cv2.rectangle(frm, (x, y), (x + w, y + h), color, 1)

                if y >= 0:
                    if x < 0:
                        x+=3
                    elif y < 0:
                        y+=3
                    face_mask = frm[y:y + h, x:x + w]
                    '''
                    Age & Gender Predict Networks
                    '''
                    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
                    blob_age_gender = cv2.dnn.blobFromImage(frm[y:y + h, x:x + w], 1.0, (227, 227), MODEL_MEAN_VALUES,
                                                            swapRB=False)
                    genderNet.setInput(blob_age_gender)
                    ageNet.setInput(blob_age_gender)
                    Gender_layerOutputs = genderNet.forward()
                    Age_layerOutputs = ageNet.forward()

                    gender = genderList[Gender_layerOutputs[0].argmax()]

                    age = ageList[Age_layerOutputs[0].argmax()]

                    cv2.imshow("face", face_mask)
                    '''
                    Face Blurring using GaussianBlur
                    '''
                    f_W, f_H = face_mask.shape[:2]
                    KW = int(f_W // 7) | 1
                    KH = int(f_H // 7) | 1
                    face_blur_mask = cv2.GaussianBlur(face_mask, (23, 23), 30)
                    cv2.imshow("blur", face_blur_mask)

                    frm[y:y + face_blur_mask.shape[0], x:x + face_blur_mask.shape[1]] = face_blur_mask

                text = "{}: {:.4f}: {}: {}".format(mask_class_names[mask_classIds[i]], mask_confidences[i], gender, age)
                cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if totalFrames % 2 == 0:
            trackers = []
            if len(yolo_idxs) > 0:
                for i in yolo_idxs.flatten():
                    (x, y) = (yolo_boxes[i][0], yolo_boxes[i][1])
                    (w, h) = (yolo_boxes[i][2], yolo_boxes[i][3])

                    color = [int(c) for c in Yolo_Box_COLORS[yolo_classIds[i]]]
                    cv2.rectangle(frm, (x, y), (x + w, y + h), color, 1)

                    # if yolo_class_names[yolo_classIds[i]] == 'person':
                    #     cv2.circle(frm, (math.floor(x + (w / 2)), math.floor(y + (h / 2))), 5, color, -1)
                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x, y, x + w, y + h)
                    tracker.start_track(rgb, rect)
                    print("rect :",rect)


                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
                    text = "{}: {:.4f}".format(yolo_class_names[yolo_classIds[i]], yolo_confidences[i])
                    cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput

        # default "% 30"
        else:
            # loop over the trackers
            print("trackers", trackers)
            for tracker in trackers:
                print("Tracking")
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

        print("rects :", rects)
        objects = ct.update(rects)
        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frm, (0, H // 2), (W, H // 2), (0, 0, 0), 3)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        print("trackers outside :", trackers)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        empty.append(totalUp)
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        empty1.append(totalDown)
                        # if the people limit exceeds over threshold, send an email alert

                        to.counted = True

                    x = []
                    # compute the sum of total people inside
                    x.append(len(empty1) - len(empty))

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frm, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frm, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        # construct a tuple of information we will be displaying on the
        info = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ("Status", status),
        ]

        info2 = [
            ("Total people inside", x),
        ]

        # Display the output
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frm, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        '''
        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frm, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        '''

        fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
        cv2.putText(frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        # Initiate a simple log to save data at end of the day
        if config.Log:
            datetimee = [datetime.datetime.now()]
            d = [datetimee, empty1, empty, x]
            export_data = zip_longest(*d, fillvalue='')

            with open('Log.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(("End Time", "In", "Out", "Total Inside"))
                wr.writerows(export_data)

        # show the output frame
        cv2.imshow("Real-Time Monitoring/Analysis Window", frm)
        key = cv2.waitKey(1) & 0xFF

        if output_video_dir != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_video_dir, fourcc, 25,
                                     (frm.shape[1], frm.shape[0]), True)

        if writer is not None:
            writer.write(frm)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

        if config.Timer:
            # Automatic timer to stop the live stream. Set to 8 hours (28800s).
            t1 = time.time()
            num_seconds = (t1 - t0)
            if num_seconds > 28800:
                break

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # # if we are not using a video file, stop the camera video stream
    # if not args.get("input", False):
    # 	vs.stop()
    #
    # # otherwise, release the video file pointer
    # else:
    # 	vs.release()

    # issue 15
    if config.Thread:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()


##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
    ##Runs for every 1 second
    # schedule.every(1).seconds.do(run)
    ##Runs at every day (9:00 am). You can change it.
    schedule.every().day.at("9:00").do(run)

    while 1:
        schedule.run_pending()

else:
    detect()
#     if not grabbed:
#         break
#     frame = imutils.resize(frame, 1440)
#     #detect(frame, mask_net, yolo_net, mask_layer, yolo_layer, totalUp, totalDown, totalFrames, x, empty, empty1)
#     detect(frame, mask_net, yolo_net, mask_layer, yolo_layer)
#
#     if enable_display == 1:
#         cv2.imshow("detections", frame)
#         fps.stop()
#


print("Done")
