import json
import math
# import joblib
from collections import OrderedDict, Counter

import cv2
import numpy as np
from keras.models import load_model

from utils.image_util import crop_image


class Detector:

    def __init__(self, models):
        self.detectors = OrderedDict()
        self.colors = ['黑', '正常', '白', '黄']
        self.init_detectors(models)

    def init_detectors(self, models):
        """
        加载诊断模型，用于诊断
        :param models: 模型名和对应路径的有序字典
        :return:
        """
        for name in models:
            path = models.get(name)
            self.detectors[name] = load_model(path)
            # if path.endswith("pkl"):
            #     self.detectors[name] = joblib.load(path)
            # else:
            #     self.detectors[name] = load_model(path)

    def detect(self, image):
        """
        舌诊
        依次经过
        tongue_detector->xxx
        :param image: ndarray对象，形状为420×420×3
        :return: 返回诊断结果
        """
        result = {}
        reimage = cv2.resize(image.copy(), (180, 180))
        reimage = reimage.reshape((1, 180, 180, 3))
        result['isTongue'], proba1 = self.detect_tongue(reimage)
        if result['isTongue'] == '无舌头':
            return result, proba1
        reimage = crop_image(image)
        result['color'], proba2 = self.detect_color(reimage)
        result['indentation'], proba3 = self.detect_indentation(reimage)
        return result, (proba1 + proba2 + proba3) / 3

    def detect_many(self, images):
        # 判断有没有舌头
        tongue_result, tongue_probability = zip(*[self.detect_tongue(image) for image in images])
        is_tongue = Counter(tongue_result).most_common()[0][0]
        if is_tongue == '无舌头':
            return {'isTongue': '无舌头'}, tongue_probability
        else:
            color_result, color_probability = zip(*[self.detect_color(image) for image in images])
            color = Counter(color_result).most_common()[0][0]
            indentation_result, indentation_probability = zip(*[self.detect_indentation(image) for image in images])
            indentation = Counter(indentation_result).most_common()[0][0]
            probability = np.mean(np.array(tongue_probability + color_probability + indentation_probability))
            return {
                       'isTongue': '有舌头',
                       'color': color,
                       'indentation': indentation
                   }, probability

    def detect_tongue(self, image):
        """
        检测图像中是否有舌头
        :param image: ndarray对象，形状为1×180×180×3或180×180×3
        :return: 如果有舌头返回True，否则返回False
        """
        if image.ndim == 3:
            image = image.reshape((1, 180, 180, 3))
        pred = self.detectors['TongueDetector'].predict(image, verbose=0)
        probability = round(abs(pred[0, 0] - 0.5) / 0.5 * 100, 2)
        return '有舌头' if pred[0, 0] >= 0.5 else '无舌头', probability

    def detect_color(self, image):
        """
        判断舌色
        :param image: ndarray对象，形状为1×180×180×3或180×180×3
        :return: 返回0，1，2，3（黑，正，白，黄）
        """
        if image.ndim == 3:
            image = image.reshape((1, 180, 180, 3))
        pred = self.detectors['ColorDetector'].predict(image, verbose=0)
        idx = np.argmax(pred)
        probability = round(abs(pred[0, idx] - 0.5) / 0.5 * 100, 2)
        return self.colors[idx], probability

    def detect_indentation(self, image):
        """
        判断齿痕
        :param image: ndarray对象，形状为1×180×180×3或180×180×3
        :return: 如果有齿痕返回True，否则返回False
        """
        if image.ndim == 3:
            image = image.reshape((1, 180, 180, 3))
        pred = self.detectors['IndentationDetector'].predict(image, verbose=0)
        probability = round(abs(pred[0, 0] - 0.5) / 0.5 * 100, 2)
        return '有齿痕' if pred[0, 0] >= 0.5 else '无齿痕', probability


if __name__ == '__main__':
    diagnosis = json.load(open("static/diagnosis.json", "r", encoding="utf-8"))
    models = OrderedDict([
        ('TongueDetector', 'static/models/tongue_detector.keras'),
        ('ColorDetector', 'static/models/color_detector.keras'),
        ('IndentationDetector', 'static/models/indentation_detector.keras')
    ])
    detector = Detector(models)
    from utils.image_util import url2image2, crop_image

    url = "https://lmg.jj20.com/up/allimg/1114/040221103339/210402103339-7-1200.jpg"
    image = url2image2(url)
    image = crop_image(image)
    cv2.imshow("img", image)
    cv2.waitKey()
    # image = cv2.resize(image, (180, 180))
    result = detector.detect_many([image])
    print(result)
