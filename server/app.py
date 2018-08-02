from flask import Flask, render_template, Response
#import cv2
import os
import glob
from flask_cors import CORS

from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import argparse
from datetime import datetime
import time
from io import BytesIO

correct = 0

app = Flask(__name__)
CORS(app)

model_path = './trained_model/frozen_inference_graph.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(model_path, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
	  (im_height, im_width, 3)).astype(np.uint8)

label_map = label_map_util.load_labelmap('./trained_model/labels.txt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

@app.route('/')
def index():
	return render_template('index.html')

def find_wally():
	global correct
	total = 0
	correct = 0
	#start_t = strftime("%Y-%m-%d %H:%M:%S", gmtime())
	t_end = time.time() + 60
	
	with detection_graph.as_default():
			with tf.Session(graph=detection_graph) as sess:
					for image in glob.glob('images/*.jpg'):
						if time.time() < t_end:
								print(image)
								image_np = load_image_into_numpy_array(Image.open(image))
								image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
								boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
								scores = detection_graph.get_tensor_by_name('detection_scores:0')
								classes = detection_graph.get_tensor_by_name('detection_classes:0')
								num_detections = detection_graph.get_tensor_by_name('num_detections:0')
								# Actual detection.
								(boxes, scores, classes, num_detections) = sess.run(
								[boxes, scores, classes, num_detections],
										feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})
								if scores[0][0] < 0.1:
										total = total + 1
										continue

								print('Wally found')
								total = total + 1
								correct = correct + 1
								vis_util.visualize_boxes_and_labels_on_image_array(
										image_np,
										np.squeeze(boxes),
										np.squeeze(classes).astype(np.int32),
										np.squeeze(scores),
										category_index,
										use_normalized_coordinates=True,
										line_thickness=8)
								#plt.figure(figsize=(12, 8))
								#plt.imshow(image_np)
								#plt.show()

								#NOTE: RESTACK IS UNESSISARY NOW THAT IM NOT USING IMAGE ENCODE FROM CV2, TAKE OUT

								#LOOK HERE
								print(np.shape(image_np))
								print(type(image_np))
								B = image_np[:,:,0]
								G = image_np[:,:,1]
								R = image_np[:,:,2]
								#print('B SHAPE:')
								#print(np.shape(B))
								RGB_img = np.dstack([B,G,R])
								#print('RGB_TSHAPE:')
								#print(np.shape(RGB_img_T))
								#RGB_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
								im = Image.fromarray(RGB_img)
								size = 300 , 200
								im_small = im.resize(size, Image.ANTIALIAS)

								with BytesIO() as f:
									im_small.save(f, format='JPEG')
									jpeg = f.getvalue()

								#ret, jpeg = cv2.imencode('.jpg', RGB_img)
								yield (b'--frame\r\n'
									b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n\r\n')
									#b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
								#plt.show()
						else:
							break
	time.sleep(45)
	correct = 0
	print(str(correct) + '/' + str(total))
	#print(start_t)
	#print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	
@app.route('/wally_feed')
def wally_feed():
	return Response(find_wally(),
					mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/AI_Correct')
def AI_Correct():
    global correct
    return Response(str(correct))
