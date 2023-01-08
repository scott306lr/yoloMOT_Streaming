import argparse
import os
import time

# # limit the number of cpus used by high performance libraries
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import subprocess as sp
from flask import Flask, request, jsonify, Response, render_template, render_template_string, session
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import random 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.ocsort.ocsort import OCSort

# remove duplicated stream handler to avoid duplicated logging
#logging.getLogger().removeHandler(logging.getLogger().handlers[0])


# global outputFrame
# global lock_frame
# global metadata

lock_frame = threading.Lock()
outputFrame = None
metadata = None

lock_users = threading.Lock()
users = []
userConfig = {}


TEMPLATE_DIR = "Frontend"#"templates"
STATIC_DIR = "Frontend"
# ASYNC_MODE = "eventlet"
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'secret!'
# app.config["DEBUG"] = True
# CORS(app, resources={r"/*": {
#     "origins": "*",
#     "Access-Control-Allow-Origin": "*"
# }})
socketio = SocketIO(app, cors_allowed_origins="*") #async_mode=ASYNC_MODE, 
# socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        name='exp',  # save results to project/name
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):


    global outputFrame
    global lock_frame
    global metadata
    global names

    source = str(source)
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Dataloader
    if webcam:
        # show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = OCSort(
            det_thresh=0.45,
            iou_threshold=0.2,
            use_byte=False 
        )
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * nr_sources

    # Run tracking
    #model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                # p = Path(p)  # to Path
                # s += f'{i}: '
                # txt_file_name = p.name
                # save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # else:
                # p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                # p = Path(p)  # to Path
                # video file
                # if source.endswith(VID_FORMATS):
                    # txt_file_name = p.stem
                    # save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                # else:
                    # txt_file_name = p.parent.name  # get folder name containing current img
                    # save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            # txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            frame_temp = im0.copy()
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            metadata_temp = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = tracker_list[i].update(det.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        metadata_temp.append({'box':bboxes, 'id':int(id), 'cls':int(cls), 'conf':conf})

                # LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                
            else:
                # LOGGER.info('No detections')
                pass

            with lock_frame:
                outputFrame, metadata = frame_temp, metadata_temp

            prev_frames[i] = curr_frames[i]


def canny_modifier(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 50, 100)
    frame = cv2.dilate(frame, None, iterations=1)
    frame = cv2.erode(frame, None, iterations=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def draw_rectangles(frame, boxes, show_class_id, confidence):
    global names
    # draw rectangles

    # draw only the passed "class id", and bboxes with "confidence" higher than specific threshold.
    annotator = Annotator(frame, line_width=2, example=str(names))
    for box in boxes:
        c = box['cls']
        if c not in show_class_id:
            continue
        if box['conf'].numpy() < confidence:
            # print("Out by confidence", box['conf'].numpy(), confidence)
            continue
        label = f"{box['id']} {names[c]} {box['conf']:.2f}"
        color = colors(c, True)
        annotator.box_label(box['box'], label, color=color)
    im0 = annotator.result()
    return im0


def draw_metadata(frame, metadata, resolution):
    # detected boxes,
    # draw metadata,
    # draw fps etc....

    if (resolution == '360p'):
        frame = cv2.resize(frame, (480, 360))

    if (resolution == '480p'):
        frame = cv2.resize(frame, (640, 480))

    return frame


def process_frame(frame, metadata, configs):
    # frame mode
    case = configs["mode"]
    if case == "original":
        pass
    elif case == "canny":
        frame = canny_modifier(frame)
    else:
        pass

    # draw rectangles
    frame = draw_rectangles(frame, metadata, configs['show_class_id'], configs['confidence'])

    # draw metadata
    frame = draw_metadata(frame, metadata, configs['resolution'])

    return frame

def generate(myID):
    global outputFrame, lock_frame, metadata, userConfig  # ,user, lock_users

    prev_frame = None
    while True:
        with lock_frame:
            if (outputFrame is None):
                continue

            # replace testConfig with userConfig[myID]
            if userConfig[myID]["play"] == True or prev_frame is None:
                processedFrame = process_frame(outputFrame, metadata, userConfig[myID])
                prev_frame = processedFrame

            (flag, encodedImage) = cv2.imencode(".jpg", prev_frame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

from uuid import uuid4

@app.route('/')
def index():
    # return the rendered template
    if request.method == "GET":
        new_user_id = str(uuid4())
        session["ID"] = new_user_id
        
        with lock_users:
            users.append(session.get('ID'))

        userConfig[session.get('ID')] = {
            "id": session.get('ID'),
            "resolution": "480p",
            "mode": "original",
            "confidence": 0.5,
            "play": True,
            "show_class_id": [ i for i in range(0, 80) ]
        }
        with open('./Frontend/index.html', 'r') as file:
            data = file.read().rstrip()
            data = data.replace('USER_DEF_ID', new_user_id)
            return render_template_string(data)
        # return render_template("index.html")

# obtain id from the front end
@app.route("/<myID>/video_feed")
def video_feed(myID):
    return Response(generate(myID), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    print(f"client with id:{session.get('ID')} has connected")
    emit("connect", {'data': f"user {session.get('ID')} connected",
                     'id': session.get('ID')}, broadcast=True)


# @socketio.on("bullet")
# def bullet(data):
#     print(f"user {session.get('ID')} fired a bullet:\n {data}")
#     emit("bullet", {'data': data, 'id': session.get('ID')}, broadcast=True)



@socketio.on("confidence")
def confidence(data):
    userConfig[session.get('ID')]["confidence"] = float(data)
    emit("confidence", {'data': data, 'id': session.get('ID')}, broadcast=False)


@socketio.on('resolution')
def change_resolution(data):
    print("data from the front end: ", str(data))
    userConfig[session.get('ID')]["resolution"] = data
    emit("resolution", {'data': data, 'id': session.get('ID')}, broadcast=False)

# play and pause
@socketio.on('play_toggle')
def play_toggle(data):
    userConfig[session.get('ID')]["play"] = False if data == 'pause' else True
    print(f"got play/pause {data}")
    emit("play_toggle", {'data': data, 'id': session.get('ID')}, broadcast=False)


@socketio.on('show_class_id')
def change_show_class_id(data):
    """event listener when client types a message"""
    print("data from the front end: ", str(data))
    userConfig[session.get('ID')]["show_class_id"] = list(map(int, data))
    emit("show_class_id", {'data': data, 'id': session.get('ID')}, broadcast=False)


@socketio.on('mode')
def change_mode(data):
    """event listener when client types a message"""
    print("data from the front end: ", str(data))
    userConfig[session.get('ID')]["mode"] = data
    emit("mode", {'data': data, 'id': session.get('ID')}, broadcast=False)


@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    with lock_users:
        users.remove(session.get('ID'))
    del userConfig[session.get('ID')]
    print(f"client with id:{session.get('ID')} has disconnected")
    emit("disconnect", {
         'data': f"user {session.get('ID')} disconnected", 'id': session.get('ID')}, broadcast=True)


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    global outputFrame
    global lock_frame
    global metadata
    global names
    t = threading.Thread(target=run, kwargs=vars(opt))
    t.daemon = True
    t.start()
    
    time.sleep(3)
    print("Opening host socket...")
    # socketio.run(app, host="0.0.0.0", debug=True, port=5000)
    socketio.run(app, host="0.0.0.0", port=12345, debug=True, use_reloader=False)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    # print_args(vars(opt))
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)