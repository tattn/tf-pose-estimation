import argparse
import logging
import time

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

from concurrent.futures import ThreadPoolExecutor
from http.server import HTTPServer, BaseHTTPRequestHandler
import json



logger = logging.getLogger('Pose Estimation Server')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)



def pose_estimation(model, zoom, camera):
    logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
    w, h = model_wh(model)
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    cam = cv2.VideoCapture(camera)
    ret_val, image = cam.read()
    logger.info('camera image=%dx%d' % (image.shape[1], image.shape[0]))

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess')
        if zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('inference')
        humans = e.inference(image)

        if cv2.waitKey(1) == 27:
            break
    logger.debug('finished+')



def run_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-type','application/json')
            self.end_headers()
            result_json = {"hoge": 1}
            self.wfile.write(json.dumps(result_json).encode('UTF-8'))

    PORT = 8080

    httpd = HTTPServer(("", PORT), Handler)
    logger.debug(f'running server on {PORT} port')
    httpd.serve_forever()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--model', type=str, default='mobilenet_thin_432x368', help='cmu_640x480 / cmu_640x360 / mobilenet_thin_432x368')
    args = parser.parse_args()

    # with ThreadPoolExecutor(max_workers=2) as executor:
        # executor.submit(run_server)
        # executor.submit(pose_estimation, args.model, args.zoom, args.camera)
    pose_estimation(args.model, args.zoom, args.camera)

