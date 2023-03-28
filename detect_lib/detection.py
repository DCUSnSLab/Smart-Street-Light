import numpy as np
import dlib

from detect_lib.set_neural_net import *

'''
Age & Gender Network Directory
'''
age_proto = "caffe/age_deploy.prototxt"
age_model = "caffe/age_net.caffemodel"

gender_proto = "caffe/gender_deploy.prototxt"
gender_model = "caffe/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

'''
YOLOv4 Parameter
'''
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.4
enable_display = 1

def maskDetection(frm, blob, mask_net, mask_layer, mask_class_names, H, W, Mask_Box_COLORS):
    mask_boxes = []
    mask_classIds = []
    mask_confidences = []

    mask_net.setInput(blob)
    Mask_layerOutputs = mask_net.forward(mask_layer)

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

    mask_idxs = cv2.dnn.NMSBoxes(mask_boxes, mask_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(mask_idxs) > 0:
        for i in mask_idxs.flatten():
            (x, y) = (mask_boxes[i][0], mask_boxes[i][1])
            (w, h) = (mask_boxes[i][2], mask_boxes[i][3])

            color_mask = [int(c) for c in Mask_Box_COLORS[mask_classIds[i]]]
            cv2.rectangle(frm, (x, y), (x + w, y + h), color_mask, 1)

            if y >= 0:
                if x < 0:
                    x += 3
                elif y < 0:
                    y += 3
                face_mask = frm[y:y + h, x:x + w]

                blob_age_gender = cv2.dnn.blobFromImage(frm[y:y + h, x:x + w], 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                ageNet, genderNet = setagegenderDetectionNet(age_model, age_proto, gender_model, gender_proto)
                age, gender = age_gender_Detection(blob_age_gender, ageNet, genderNet)

                faceFiltering(frm, face_mask, x, y)

            if mask_class_names[mask_classIds[i]] == "no_mask":
                text = "{}: {:.4f}: {}: {}".format(mask_class_names[mask_classIds[i]], mask_confidences[i], gender, age)
            elif mask_class_names[mask_classIds[i]] == "mask":
                text = "{}: {:.4f}".format(mask_class_names[mask_classIds[i]], mask_confidences[i])
            cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_mask, 1)

def yoloDetection_with_setTracker(frm, blob, yolo_net, yolo_layer, yolo_class_names, H, W, Yolo_Box_COLORS,
                                  ct, totalFrames, trackers, rects):
    yolo_boxes = []
    yolo_classIds = []
    yolo_confidences = []
    rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)

    yolo_net.setInput(blob)
    Yolo_layerOutputs = yolo_net.forward(yolo_layer)

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

    yolo_idxs = cv2.dnn.NMSBoxes(yolo_boxes, yolo_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if totalFrames % 2 == 0:
        trackers.clear()
        if len(yolo_idxs) > 0:
            for i in yolo_idxs.flatten():
                (x, y) = (yolo_boxes[i][0], yolo_boxes[i][1])
                (w, h) = (yolo_boxes[i][2], yolo_boxes[i][3])

                color_yolo = [int(c) for c in Yolo_Box_COLORS[yolo_classIds[i]]]
                cv2.rectangle(frm, (x, y), (x + w, y + h), color_yolo, 1)

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, x + w, y + h)
                tracker.start_track(rgb, rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                trackers.append(tracker)
                text = "{}: {:.4f}".format(yolo_class_names[yolo_classIds[i]], yolo_confidences[i])
                cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yolo, 1)

    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    # default "% 30"
    else:
        if len(yolo_idxs) > 0:
            for i in yolo_idxs.flatten():
                (x, y) = (yolo_boxes[i][0], yolo_boxes[i][1])
                (w, h) = (yolo_boxes[i][2], yolo_boxes[i][3])

                color_yolo = [int(c) for c in Yolo_Box_COLORS[yolo_classIds[i]]]
                cv2.rectangle(frm, (x, y), (x + w, y + h), color_yolo, 1)

                text = "{}: {:.4f}".format(yolo_class_names[yolo_classIds[i]], yolo_confidences[i])
                cv2.putText(frm, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_yolo, 1)

        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
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

    objects = ct.update(rects)
    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' or 'down'
    #cv2.line(frm, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
    return objects, status

def age_gender_Detection(blob_age_gender, ageNet, genderNet):
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    '''
    Age & Gender Predict Networks
    '''
    genderNet.setInput(blob_age_gender)
    ageNet.setInput(blob_age_gender)

    Gender_layerOutputs = genderNet.forward()
    Age_layerOutputs = ageNet.forward()

    gender = genderList[Gender_layerOutputs[0].argmax()]

    age = ageList[Age_layerOutputs[0].argmax()]

    return age, gender

def faceFiltering(frm, face_mask, x, y):
    '''
    Face Blurring using GaussianBlur
    '''
    face_blur_mask = cv2.GaussianBlur(face_mask, (23, 23), 30)
    frm[y:y + face_blur_mask.shape[0], x:x + face_blur_mask.shape[1]] = face_blur_mask

    cv2.imshow("face", face_mask)
    cv2.imshow("blur", face_blur_mask)

