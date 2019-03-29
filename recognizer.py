import cv2
import base64
import requests
import json
from darknet.darkflow.net.build import TFNet

options = {"model": "darknet/cfg/yolov2_coco.cfg", "load": "darknet/bin/yolov2_coco.weights", "threshold": 0.1}

tfnet = TFNet(options)

API_KEY = "AIzaSyCtXboroGSfXE5jiT6zp7AZbLVT1C7t91I"
URL = "https://vision.googleapis.com/v1/images:annotate?&key=" + API_KEY

print(URL)

def get_middle_point(info):
    res = ((int(info['boundingPoly']["vertices"][0]['x'] + info['boundingPoly']["vertices"][-2]['x'])/2),
           (int(info['boundingPoly']["vertices"][0]['y'] + info['boundingPoly']["vertices"][-2]['y'])/2))
    return res, (info['boundingPoly']["vertices"][0]['x'],info['boundingPoly']["vertices"][0]['y'] ), \
           (info['boundingPoly']["vertices"][-2]['x'], info['boundingPoly']["vertices"][-2]['y'])


def get_text(img, topleft, bottomright):
    retval, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode("utf-8")
    req = {
        "requests": [
            {
                "features": [{
                    "type": "LOGO_DETECTION"
                }
                ],
                "image": {
                    "content": jpg_as_text
                }

            }
        ]
    }
    res = requests.post(URL, data=json.dumps(req)).json()
    print(res)
    for v in res["responses"]:
        if "logoAnnotations" in v:
            for info in v["logoAnnotations"]:
                mid, topleft_text, bottomright_text = get_middle_point(info)
                if topleft[0]<=mid[0]<=bottomright[0] and topleft[1]<=mid[1]<=bottomright[1]:
                    img = cv2.rectangle(img, topleft_text, bottomright_text, (0, 255, 0))
                    img = cv2.putText(img, info["description"], bottomright_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

    return img


def detect_cup(img):
    result = tfnet.return_predict(img)
    for v in result:
        if v['label'] == 'cup':
            img = cv2.rectangle(img, (v['topleft']['x'], v['topleft']['y']), (v['bottomright']['x'],v['bottomright']['y']), (255, 0, 0))
            img = cv2.putText(img, 'cup', (v['bottomright']['x'],v['bottomright']['y']), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

            return img, (v['topleft']['x'],v['topleft']['y'] ), (v['bottomright']['x'], v['bottomright']['y'])
    return None, None, None


def recognize(path_to_video):
    cap = cv2.VideoCapture(path_to_video)
    i = 0

    while True:
        ret, frame = cap.read()
        if i % 3 == 0:
            # cv2.imwrite("frames/" + str(i) + ".jpg", frame)
            if not ret or frame is None:
                break
            cup, topleft, bottomright = detect_cup(frame)
            if cup is not None:
                img = get_text(frame, topleft, bottomright)
                cv2.imshow("cup", img)
                cv2.waitKey(1)
            else:
                cv2.imshow("cup", frame)
                cv2.waitKey(1)
        print(i)
        i += 1


if __name__ == '__main__':
    # implicit()
    # test_api_2("video/test.ogv")
    recognize("video/test.ogv")
