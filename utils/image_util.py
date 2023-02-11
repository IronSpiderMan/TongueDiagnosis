import io

import cv2
import requests
import numpy as np
from PIL import Image


def url2image(url: str) -> np.ndarray:
    capture = cv2.VideoCapture(url)
    _, image = capture.read()
    return image


def url2image2(url: str) -> np.ndarray:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers)
    try:
        image = Image.open(io.BytesIO(resp.content))
        image = np.array(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(resp)
        print(e)
        return None


def crop_image(image: np.ndarray) -> np.ndarray:
    # 把图片缩放到n×240
    h, w, _ = image.shape
    fx = round(240 / w, 2)
    image = cv2.resize(image, (0, 0), fx=fx, fy=fx)
    nh, nw, _ = image.shape
    # 截取图片中心180×180区域
    x = (nw - 180) // 2
    y = (nh - 180) // 2
    image = Image.fromarray(image)
    image = image.crop((x, y, x + 180, y + 180))
    return np.array(image)


if __name__ == '__main__':
    path = "https://img2.baidu.com/it/u=272103827,29441899&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=625"
    url2image2(path)
    # path = r"C:\Users\86185\Desktop\111.jpg"
    # image = cv2.imread(path)
    # image = crop_image(image)
    # print(image.shape)
    # cv2.imshow("img", image)
    # cv2.waitKey()
