import argparse

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import subprocess as sp
from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading

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
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
#logging.getLogger().removeHandler(logging.getLogger().handlers[0])


global outputFrame
global lock_frame
global metadata
lock_frame = threading.Lock()
outputFrame = None
metadata = None
lock_users = threading.Lock()
users = []
userConfig = {}
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):


    global outputFrame
    global lock_frame
    global metadata
    global names
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
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
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(nr_sources):
        tracker = create_tracker(tracking_method, reid_weights, device, half)
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
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
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
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            frame_temp = im0.copy()
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

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
                
                metadata_temp = []
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        metadata_temp.append({'box':bboxes, 'id':id, 'cls':int(cls), 'conf':conf})

                LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                
            else:
                LOGGER.info('No detections')

            with lock_frame:
                outputFrame, metadata = frame_temp, metadata_temp

            prev_frames[i] = curr_frames[i]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def canny_modifier(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 50, 100)
    frame = cv2.dilate(frame, None, iterations=1)
    frame = cv2.erode(frame, None, iterations=1)
    return frame

#! To be implemented


def draw_rectangles(frame, boxes, show_class_id):
    global names
    # draw rectangles

    # draw only the passed "class id", and bboxes with "confidence" higher than specific threshold.
    annotator = Annotator(frame, line_width=2, example=str(names))
    for box in boxes:
        c = box['cls']
        if c not in show_class_id:
            continue
        label = f"{box['id']} {names[c]} {box['conf']:.2f}"
        color = colors(c, True)
        annotator.box_label(box['box'], label, color=color)
    im0 = annotator.result()
    
    return im0

#! To be implemented


def draw_metadata(frame, metadata):
    # detected boxes,
    # draw metadata,
    # draw fps etc....
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
    frame = draw_rectangles(frame, metadata, configs['show_class_id'])

    # draw metadata
    frame = draw_metadata(frame, metadata)

    return frame


def generate(myID):
    global outputFrame, lock_frame, metadata, userConfig  # ,user, lock_users
    testConfig = {
        "id": myID,
        "resolution": "640x480",
        "mode": "original",
        "play": True,
        "show_class_id": [0, 1, 3]  # need to draw class id == 0, 1, 3
    }

    while True:
        with lock_frame:
            if (outputFrame is None):
                continue

            # replace testConfig with userConfig[myID]
            processedFrame = process_frame(outputFrame, metadata, testConfig)

            (flag, encodedImage) = cv2.imencode(".jpg", processedFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/", methods=['GET', 'POST'])
def index():
    # return the rendered template
    if request.method == "GET":
        return render_template("index.html")

# obtain id from the front end
@app.route("/<myID>/video_feed")
def video_feed(myID):
    return Response(generate(myID), mimetype="multipart/x-mixed-replace; boundary=frame")


@socketio.on("connect")
def connected():
    """event listener when client connects to the server"""
    with lock_users:
        users.append(request.sid)
    userConfig[request.sid] = {
        "id": request.sid,
        "resolution": "640x480",
        "mode": "original",
        "play": True,
        "show_class_id": [1 for i in range(80)],
    }

    print(f"client with id:{request.sid} has connected")
    emit("connect", {'data': f"user {request.sid} connected",
                     'id': request.sid}, broadcast=True)


@socketio.on("bullet")
def bullet(data):
    """event listener when client types a message"""
    print(f"user {request.sid} fired a bullet:\n {data}")
    emit("bullet", {'data': data, 'id': request.sid}, broadcast=True)


# @socketio.on('resolution')
# def change_resolution(data):
#     """event listener when client types a message"""
#     print("data from the front end: ", str(data))
#     userConfig[request.sid]["resolution"] = data
#     emit("resolution", {'data': data, 'id': request.sid}, broadcast=False)


@socketio.on('play_toggle')
def play_toggle():
    new_state = not userConfig[request.sid]["play"]
    userConfig[request.sid]["play"] = new_state
    emit("play", {'data': new_state, 'id': request.sid}, broadcast=False)


@socketio.on('show_class_id')
def change_show_class_id(data):
    """event listener when client types a message"""
    print("data from the front end: ", str(data))
    userConfig[request.sid]["show_class_id"] = data
    emit("show_class_id", {'data': data, 'id': request.sid}, broadcast=False)


@socketio.on('mode')
def change_mode(data):
    """event listener when client types a message"""
    print("data from the front end: ", str(data))
    userConfig[request.sid]["mode"] = data
    emit("mode", {'data': data, 'id': request.sid}, broadcast=False)


@socketio.on("disconnect")
def disconnected():
    """event listener when client disconnects to the server"""
    with lock_users:
        users.remove(request.sid)
    del userConfig[request.sid]
    print(f"client with id:{request.sid} has disconnected")
    emit("disconnect", {
         'data': f"user {request.sid} disconnected", 'id': request.sid}, broadcast=True)



def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    global outputFrame
    global lock_frame
    global metadata
    global names
    t = threading.Thread(target=run, kwargs=vars(opt))
    t.daemon = True
    t.start()
    # socketio.run(app, host="0.0.0.0", debug=True, port=5000)
    app.run(host="0.0.0.0", port="12345", debug=True,
            threaded=True, use_reloader=False)
    # run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)