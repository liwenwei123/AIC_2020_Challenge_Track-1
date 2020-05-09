from collections import deque
from sort import *
import math
import csv
import cv2
import math
import sys
import time
from get_lines import *
import torch
import os

import argparse
from sys import platform

from models import *  
from utils.datasets import *
from utils.utils import *
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    
    r = max(new_shape) / max(shape)
    if not scaleup:  
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] 
    if auto:  
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  

    dw /= 2 
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp) 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  
    return img, ratio, (dw, dh)

def get_boxes(frame):

    img = letterbox(frame, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1) 
    img = np.ascontiguousarray(img)
    
    img = torch.from_numpy(img).to(device)
    img = img.float()  
    img /= 255.0  
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=augment)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres,
                               multi_label=False, classes=None, agnostic=agnostic_nms)
    # Process detections
    for i, det in enumerate(pred): 
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            bboxes = det.cpu().numpy()
            return bboxes
# ----------------------------------------------------------------------------------------------------------------------

# dataset path
data_path='data'
datasetA_path = os.path.join(data_path, 'Dataset_A')
video_id_dict = get_video_id(datasetA_path)
classes = {0:'car',1:'truck'}

# ----------------------------------------------------------------------------------------------------------------------

agnostic_nms=False
augment=False
cfg=os.path.join('cfg', 'yolov3.cfg')
conf_thres=0.3
device=''
img_size=512
iou_thres=0.6
output_path='output'
weights = os.path.join('weights', 'best.pt')
device = torch_utils.select_device(device)

model = Darknet(cfg, img_size)
attempt_download(weights)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()


# input: start_video_id end_video_id 
start_video = sys.argv[1]
end_video = sys.argv[2]

   
for video_id in range(int(start_video),int(end_video)+1):

    print('video_id:%s' % str(video_id))
    video_name = video_id_dict[video_id]  
    cam_id = int(video_name.split('.')[0].split('_')[1])
    mov_nums, lines, directions, mov_rois = get_lines(cam_id)
    roi_nums, rois = get_rois(cam_id, data_path)

    counts = [0] * mov_nums
    counts_roi = [0] * roi_nums

    vs = cv2.VideoCapture(os.path.join(datasetA_path,video_name))

    (W, H) = (None, None)
    writer = None

# ----------------------------------------------------------------------------------------------------------------------

    last_frames = {}
    tracker = Sort()
    memory = {}
    pts = [deque(maxlen=50) for _ in range(1000000)]
    detect_flag = False

    flags = [False] * mov_nums

    indexids = [0] * mov_nums

    delays = []

    for i in range(mov_nums):
        delays.append([])

# ----------------------------------------------------------------------------------------------------------------------

    # save output result of every video
    csv_file_processed = open(os.path.join('.', output_path, '{}.csv'.format(video_id)), 'w')
    csv_writer_processed = csv.writer(csv_file_processed)
    csv_writer_processed.writerow(['video_id', 'frame_id', 'movement_id', 'vehicle_class_id'])

    data = {}
    frame_count = 0

    while True:

        ret, frame = vs.read()

        if not ret:
            result = []
            result_ori = []
            for key in data:
                video_id, frame, mov, name, roi_flag = data[key][0], data[key][1],data[key][2],data[key][3],data[key][4]
                if name != name:
                    name = 1
                result_ori.append((str(video_id), str(int(frame)), str(mov + 1), str(name)))
                if roi_flag==True:
                    if int(frame)>frame_count:
                        frame=frame_count
                    result.append((str(video_id), str(int(frame)), str(mov + 1), str(name)))
                else:
     
                    if len(delays[mov])>0:
                        frame_delay = frame + sum(delays[mov])/len(delays[mov])
                    else:
                        frame_delay = last_frames[key]
       
                    if int(frame_delay)>frame_count:
                        frame_delay=frame_count

                    result.append((str(video_id), str(int(frame_delay)), str(mov + 1), str(name)))
            csv_writer_processed.writerows(result)
            csv_file_processed.close()

            break

        frame_count += 1
        print(frame_count)

# ----------------------------------------------------------------------------------------------------------------------

        with torch.no_grad():
            bboxes = get_boxes(frame)

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        boxes = []
        confidences = []
        classIDs = []

        try:
            for i, bbox in enumerate(bboxes):
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])

                boxes.append([int(coor[0]), int(coor[1]), int(coor[2] - coor[0]), int(coor[3] - coor[1]), class_ind])
                confidences.append(float(score))
                classIDs.append(class_ind)

        except:
            print(frame_count)

        idxs = list(range(len(boxes)))
        dets = []
        record = []

        if len(idxs) > 0:
            for i in idxs:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                class_ind = bboxes[i][5]
                if classes[class_ind] == 'car' or \
                        classes[class_ind] == 'bus' or \
                        classes[class_ind] == 'truck' or \
                        classes[class_ind] == 'train':
                    dets.append([x, y, x + w, y + h, confidences[i]])
                    center_x = int(x + 0.5 * w)
                    cneter_y = int(y + 0.5 * h)
                    record.append((center_x, cneter_y, int(class_ind)))

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        #print(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}

# ------------------------------------------------------------------------------------------------------------------

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]



        if len(boxes) > 0:
            i = int(0)
            for box in boxes:

                if indexIDs[i] in last_frames:
                    if frame_count > last_frames[indexIDs[i]]:
                        last_frames[indexIDs[i]]=frame_count

                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                center = (int(0),int(0))
                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                    p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))

# -------------------------------------------------------------------------------------------------------------------
                    for mov in range(mov_nums):
                        if intersect(p0, p1, lines[mov][0], lines[mov][1]):
                            if directions == []:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center=p0
                                break
                            if directions[mov] == 1 and x2 < x:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 2 and x2 > x:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 3 and y2 < y:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break
                            if directions[mov] == 4 and y2 > y:
                                detect_flag = True
                                flags[mov] = True
                                indexids[mov] = indexIDs[i]
                                last_frames[indexIDs[i]] = frame_count
                                center = p0
                                break

                    for roi in range(roi_nums):

                        if intersect(p0, p1, rois[roi][0], rois[roi][1]):

                            if indexIDs[i] in data.keys():
                                delays[data[indexIDs[i]][2]].append(frame_count - data[indexIDs[i]][1])
                                data[indexIDs[i]][1] = frame_count
                                data[indexIDs[i]][4] = True
                               

                i += 1
                if detect_flag:
                    name = '1'

                    for x in record:
                        d1 = x[0] - center[0]
                        d2 = x[1] - center[1]
                        dis = math.sqrt(d1 * d1 + d2 * d2)
                        if dis<10:
                            name=classes[x[2]]
                            if name == 'car' or name == 'bus':
                                name = 1
                            if name == 'truck' or name == 'train':
                                name = 2

# ---------------------------------------------------------------------------------------------------------------

                for mov in range(mov_nums):
                    if flags[mov]:
                        counts[mov] += 1
                        roi_flag = False
                        data[indexids[mov]] = [str(video_id), frame_count, mov, name, roi_flag]
                        break


                detect_flag = False
                for mov in range(mov_nums):
                    flags[mov] = False

