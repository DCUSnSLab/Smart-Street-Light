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
import schedule, csv
import time, datetime

from tracker_lib.centroidtracker import CentroidTracker
from tracker_lib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from tracker_lib.mailer import Mailer
from tracker_lib import config, thread
from itertools import zip_longest
from detect_lib.detection import *

t0 = time.time()

'''
Video Directory or RTSP IP
'''
input_video_dir = "./videos/cctv_ai_hub.mp4"
# input_video_dir = "rtsp://admin:SnSlab20121208!@192.168.0.79:554/profile2/media.smp"
output_video_dir = "./result/cctv_ai_hub.mp4"

'''
print network directory
'''
print("You are now using Mask Predict File\n\t{} mask_weights\n\t{} mask_cfg\n\t{} mask_labels.".format(mask_weights,
                                                                                                        mask_cfg,
                                                                                                        mask_labels))
print("You are now using YOLO Predict File\n\t{} yolo_weights\n\t{} yolo_cfg\n\t{} yolo_labels.".format(yolo_weights,
                                                                                                        yolo_cfg,
                                                                                                        yolo_labels))
print("You are now using AGE Predict File\n\t{} age_proto\n\t{} age_model.".format(age_proto, age_model))
print("You are now using GENDER Predict File\n\t{} gender_proto\n\t{} gender_model.".format(gender_proto, gender_model))

def detect():
    '''
    If you wanna using camera change the annotation
    '''
    # vc = cv2.VideoCapture(0)

    '''
    If you wanna use IP Camera for Stream change the annotation
    and change the config.py
    '''
    #vc = VideoStream(config.url).start()
    #time.sleep(2.0)

    vc = cv2.VideoCapture(input_video_dir)

    writer = None
    write_fps = 20

    '''
    Set to Network
    '''
    mask_net, mask_layer, mask_class_names = setmaskDetectionNet()
    yolo_net, yolo_layer, yolo_class_names = setyoloDetectionNet()
    ageNet, genderNet = setagegenderDetectionNet(age_model, age_proto, gender_model, gender_proto)

    '''
    Set to Bobx color
    '''
    Yolo_Box_COLORS = np.random.randint(0, 255, size=(len(yolo_class_names), 3), dtype="uint8")
    Mask_Box_COLORS = np.random.randint(0, 255, size=(len(mask_class_names), 3), dtype="uint8")


    '''
    Tracker Variable
    '''
    ct = CentroidTracker(maxDisappeared=30, maxDistance=150)
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
        #frm = cv2.resize(frm, (1280, 720))

        (H, W) = frm.shape[:2]

        rects = []
        status = "Waiting"

        blob = cv2.dnn.blobFromImage(frm, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        start_time = time.time()
        maskDetection(frm, blob, mask_net, mask_layer, mask_class_names, H, W, Mask_Box_COLORS,)
        objects, status = yoloDetection_with_setTracker(frm, blob, yolo_net, yolo_layer, yolo_class_names, H, W,
                                      Yolo_Box_COLORS, ct, totalFrames, trackers, rects)
        end_time = time.time()

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
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
                        #print(empty1[-1])
                        # if the people limit exceeds over threshold, send an email alert
                        if sum(x) >= config.Threshold:
                            cv2.putText(frm, "-ALERT: People limit exceeded-", (10, frm.shape[0] - 80),
                                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                            if config.ALERT:
                                print("[INFO] Sending email alert..")
                                Mailer().send(config.MAIL)
                                print("[INFO] Alert sent")
                        to.counted = True

                    x = []
                    # compute the sum of total people
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
            ("Total people", totalUp + totalDown),
        ]

        # Display the output
        '''
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frm, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        '''
        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(frm, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        #fps_label = "FPS: %.2f" % (1 / (end_time - start_time))
        #cv2.putText(frm, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

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
            writer = cv2.VideoWriter(output_video_dir, fourcc, write_fps,
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
    schedule.every().day.at("9:00").do(detect)

    while 1:
        schedule.run_pending()

else:
    detect()

print("Done")
