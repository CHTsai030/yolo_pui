"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

#from flask import Flask
from flask import Response
from flask import Flask
from flask import render_template
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import threading



outputFrame = None
lock = threading.Lock()
reText = None
app = Flask(__name__)
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input", required=True,
#    help="path to input video")

args = vars(ap.parse_args())
vs = cv2.VideoCapture("videos/test.mp4")  # 本機影片
#vs = VideoStream(src=0).start()  # 攝影機

time.sleep(2.0)
## Make the WSGI interface available at the top level so wfastcgi can get it.
#wsgi_app = app.wsgi_app

# load the COCO class labels our YOLO model was trained on
#labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
labelsPath = os.path.sep.join(["yolo-coco", "voc.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(["yolo-coco", "yolov3-tiny_test7.weights"])
#configPath = os.path.sep.join(["yolo-coco", "yolov3-tiny.cfg"])
configPath = os.path.sep.join(["yolo-coco", "yolov3-tiny_test.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

##vs = cv2.VideoCapture(args["input"])
#writer = None


try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the totalnumber of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock, reText
	(W, H) = (None, None)
	# initialize the motion detector and the total number of frames
	# read thus far

	total = 0
	# loop over frames from the video stream
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it

		ret, frame = vs.read()
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		if W is None or H is None:
			(H, W) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()

        # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

        # loop over each of the layer outputs
		for output in layerOutputs:
            # loop over each of the detections
			for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]


                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
				if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")


                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])

        # ensure at least one detection exists
		if len(idxs) > 0:	
            # loop over the indexes we are keeping
			for i in idxs.flatten():
                # extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				#text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				#	confidences[i])
				reText = "{}".format(LABELS[classIDs[i]])
				cv2.putText(frame, reText, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		else:
			reText = "nothing"
		# acquire the lock, set the output frame, and release the
		# lock
		#print(type(reText))
		with lock:
			outputFrame = frame.copy()
				
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/get_text")
def get_text():
    #str ="hi"
	global reText
	return Response(reText)

#vs.release()
if __name__ == '__main__':
	import os
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()	
	
	ap.add_argument("-c", "--confidence", type=float, default=0.6,
		help="minimum probability to filter weak detections")
	ap.add_argument("-t", "--threshold", type=float, default=0.3,
		help="threshold when applying non-maxima suppression")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
			help="# of frames used to construct the background model")
	
	args = vars(ap.parse_args())
	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()
	# start the flask app
	HOST = os.environ.get('SERVER_HOST', 'localhost')
	try:
		PORT = int(os.environ.get('SERVER_PORT', '5555'))
	except ValueError:
		PORT = 5555
	app.run(HOST, PORT)
# release the video stream pointer
vs.release()
cv2.destroyAllWindows()
