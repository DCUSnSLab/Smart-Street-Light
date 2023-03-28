import cv2
import glob

'''
Neural Networks Directory
'''
mask_weights = glob.glob("yolo_mask/*.weights")[0]
mask_labels = glob.glob("yolo_mask/*.txt")[0]
mask_cfg = glob.glob("yolo_mask/*.cfg")[0]

yolo_weights = glob.glob("yolo/yolov4.weights")[0]
yolo_labels = glob.glob("yolo/labels.txt")[0]
yolo_cfg = glob.glob("yolo/yolov4.cfg")[0]

def setmaskDetectionNet():
    mask_class_names = list()

    with open(mask_labels, "r") as f:
        mask_class_names = [cname.strip() for cname in f.readlines()]

    '''
    Create Mask Predict Network
    '''
    mask_net = cv2.dnn.readNetFromDarknet(mask_cfg, mask_weights)
    mask_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    mask_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    mask_layer = mask_net.getLayerNames()
    mask_layer = [mask_layer[i[0] - 1] for i in mask_net.getUnconnectedOutLayers()]

    return mask_net, mask_layer, mask_class_names

def setyoloDetectionNet():
    yolo_class_names = list()

    with open(yolo_labels, "r") as f:
        yolo_class_names = [cname.strip() for cname in f.readlines()]

    '''
    Create YOLO Predict Network
    '''
    yolo_net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    yolo_layer = yolo_net.getLayerNames()
    yolo_layer = [yolo_layer[i[0] - 1] for i in yolo_net.getUnconnectedOutLayers()]

    return yolo_net, yolo_layer, yolo_class_names

def setagegenderDetectionNet(age_model, age_proto, gender_model, gender_proto):
    '''
    Create Age & Gender Predict Network
    '''
    ageNet = cv2.dnn.readNet(age_model, age_proto)
    genderNet = cv2.dnn.readNet(gender_model, gender_proto)

    return ageNet, genderNet


