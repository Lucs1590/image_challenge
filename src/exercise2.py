import cv2
import glob
import dlib

from os import path
from mtcnn import MTCNN
from natsort import natsorted
from matplotlib import pyplot as plt


""" model = dlib.cnn_face_detection_model_v1(
    "resources/model/mmod_human_face_detector.dat")
facesDetectadas = model(imagem, 2)
print(facesDetectadas)
print("Faces detectadas: ", len(facesDetectadas))
for face in facesDetectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(
        face.rect.right()), int(face.rect.bottom()), face.confidence)
    print(c)
    cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)

cv2.imshow("Detector CNN", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows() """


def detect_faces(_path):
    pictures = glob.glob(path.join(_path, "*.jpg")).copy()
    pictures = natsorted(pictures)
    model = MTCNN()

    for _file in pictures:
        img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
        detected_face = model.detect_faces(img)
        if detected_face:
            # plot_poits(img, detected_face)
            count_detected = {i: detected.count(i) for i in detected}
            print("MTCNN: {0} faces".format(count_detected))
            print("CNN: {0} faces".format(count_detected))

    return count_detected


def plot_poits(_image, detected_face):
    if len(detected_face):
        x1, y1, x2, y2 = detected_face[0]["box"]
        _image = cv2.rectangle(_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        for point in detected_face[0]["keypoints"].values():
            x, y = point
            _image = cv2.circle(_image, (x, y), radius=1,
                                color=(0, 0, 255), thickness=3)
    return _image


if __name__ == "__main__":
    detect_faces(
        "/home/brito/Documentos/Dev/Image_Challenge/resources/images/groups")
