import json
import random
from collections import OrderedDict

import config
from gevent import pywsgi
import numpy as np
from flask import Flask, request
from detector import Detector
from utils.image_util import url2image2, crop_image

diagnosis = json.load(open("static/diagnosis.json", "r", encoding="utf-8"))
models = OrderedDict([
    ('TongueDetector', 'static/models/tongue_detector.keras'),
    ('ColorDetector', 'static/models/color_detector.keras'),
    ('IndentationDetector', 'static/models/indentation_detector.keras')
])
detector = Detector(models)
app = Flask(__name__)
app.config.from_object(config.ProductionConfig())


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/match', methods=['GET', 'POST'])
def match():
    url = request.values.get('url')
    image = url2image2(url)
    if not isinstance(image, np.ndarray):
        return json.dumps({
            "status": "failed",
            "msg": "图片加载失败"
        }, ensure_ascii=False)
    else:
        # 截取舌头部分（取中间区域）
        # image = crop_image(image)
        result, probability = detector.detect(image)
        if result['isTongue'] == '无舌头':
            msg = diagnosis[result['isTongue']]
            msg['matchRatio'] = probability
            return json.dumps(msg, ensure_ascii=False)
        else:
            msg = diagnosis[result['isTongue']][result['color']][result['indentation']]
            if isinstance(msg['describe'], list):
                idx = random.randint(0, len(msg['describe']) - 1)
                msg['describe'] = msg['describe'][idx]
                msg['recommend'] = msg['recommend'][idx]
            msg['matchRatio'] = probability
            return json.dumps(msg, ensure_ascii=False)


@app.route('/matchMany', methods=['GET', 'POST'])
def match_many():
    urls = []
    if request.method == "GET":
        urls = request.args.get("urls").split(",")
    elif request.method == "POST":
        if request.content_type.startswith('application/json'):
            urls = request.json.get('urls').split(",")
        elif request.content_type.startswith('multipart/form-data'):
            urls = request.form.get('urls')
        else:
            urls = request.values.get("urls")
    urls = urls.split(",")
    images = [url2image2(url) for url in urls]
    flag = True
    for image in images:
        if not isinstance(image, np.ndarray):
            flag = False
        # if None in images:
        return json.dumps({
            "status": "failed",
            "msg": "图片加载失败"
        }, ensure_ascii=False)
    else:
        # 截取舌头部分（取中间区域）
        images = [crop_image(image) for image in images]
        result, probability = detector.detect_many(images)
        if result['isTongue'] == '无舌头':
            msg = diagnosis[result['isTongue']]
            msg['matchRatio'] = probability
            return json.dumps(msg, ensure_ascii=False)
        else:
            msg = diagnosis[result['isTongue']][result['color']][result['indentation']]
            if isinstance(msg['describe'], list):
                idx = random.randint(0, len(msg['describe']) - 1)
                msg['describe'] = msg['describe'][idx]
                msg['recommend'] = msg['recommend'][idx]
            msg['matchRatio'] = probability
            return json.dumps(msg, ensure_ascii=False)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 8000), app)
    server.serve_forever()
    # app.run(host='127.0.0.1', port=8000)
