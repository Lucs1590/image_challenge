import cv2
import glob
import dlib

from os import path
from mtcnn import MTCNN
from natsort import natsorted
from matplotlib import pyplot as plt


"""
print("Faces detectadas: ", len(facesDetectadas))
for face in facesDetectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(
        face.rect.right()), int(face.rect.bottom()), face.confidence)
    print(c)
    cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)

cv2.imshow("Detector CNN", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


def detect_faces(_path):
    pictures = glob.glob(path.join(_path, "*.jpg")).copy()
    pictures = natsorted(pictures)
    mtcnn_model = MTCNN()
    cnn_model = dlib.cnn_face_detection_model_v1(
        "resources/model/mmod_human_face_detector.dat")

    for _file in pictures:
        print(_file)
        mtcnn_detect(mtcnn_model, _file)
        cnn_detect(cnn_model, _file)

    return


def mtcnn_detect(model, _file):
    """
    # MTCNN Detection
    This function does face detection with MTCNN.
    """
    img = cv2.cvtColor(cv2.imread(_file), cv2.COLOR_BGR2RGB)
    detected_faces = model.detect_faces(img)
    if detected_faces:
        print("MTCNN: {0} faces".format(len(detected_faces)))
    return len(detected_faces) if detected_faces else 0


def cnn_detect(model, _file):
    """
    # CNN Detection
    This function does face detection with CNN and dlib.
    """
    img = cv2.imread(_file)
    detected_faces = model(img, 2)
    if detected_faces:
        print("CNN: {0} faces".format(len(detected_faces)))
    return len(detected_faces) if detected_faces else 0


if __name__ == "__main__":
    detect_faces(
        "/home/brito/Documentos/Dev/Image_Challenge/resources/images/groups")
