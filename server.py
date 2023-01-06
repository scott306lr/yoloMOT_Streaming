from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS

import threading
import cv2

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


def canny_modifier(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame, 50, 100)
    frame = cv2.dilate(frame, None, iterations=1)
    frame = cv2.erode(frame, None, iterations=1)
    return frame

#! To be implemented


def draw_rectangles(frame, boxes, show_class_id):
    # draw rectangles
    for i in range(len(show_class_id)):
        if show_class_id[i] == 1:
            for box in boxes[i]:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame

#! To be implemented


def draw_metadata(frame, metadata):
    # draw metadata
    for i in range(len(metadata["boxes"])):
        for box in metadata["boxes"][i]:
            x1, y1, x2, y2 = box
            cv2.putText(frame, metadata["class_names"][i], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
    frame = draw_rectangles(frame, metadata["boxes"], configs["show_class_id"])

    # draw metadata
    frame = draw_metadata(frame, metadata)

    return frame


def generate(myID):
    global outputFrame, lock_frame, metadata, userConfig  # ,user, lock_users

    while True:
        with lock_frame:
            if (outputFrame is None):
                continue

            processedFrame = process_frame(outputFrame, userConfig[myID])

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
@app.route("/video_feed")
def video_feed():
    return Response(generate(request.args.get("id")), mimetype="multipart/x-mixed-replace; boundary=frame")


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


def stream():
    global outputFrame, lock_frame, metadata

    vid = cv2.VideoCapture(0)
    while (True):
        ret, frame = vid.read()

        with lock_frame:
            outputFrame, metadata = frame, None  # run_model(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    t = threading.Thread(target=stream)
    t.daemon = True
    t.start()

    socketio.run(app, debug=True, port=5001)
    app.run(host="0.0.0.0", port="1234", debug=True,
            threaded=True, use_reloader=False)
