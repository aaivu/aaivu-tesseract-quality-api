import cv2
# import the necessary packages
import numpy as np


class FaceDetector:
  def __init__(self, image_file=None, img=None,
      prototxt_path='deploy.prototxt.txt',
      caffee_model_path='res10_300x300_ssd_iter_140000.caffemodel',
      confidence_level=0.5):
    self.img = None
    if image_file is not None:
      self.img = cv2.imread(image_file)
    elif img is not None:
      self.img = img
    else:
      "Please Provide source file as image file or image"
    self.prototxt_path = prototxt_path
    self.caffee_model_path = caffee_model_path
    self.confidence_level = confidence_level

  def detect_detections(self, image, prototxt_path, caffee_model_path):
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffee_model_path)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    return detections

  def filter_detections(self, image, confidence_level, detections, pad=20):
    faces = []
    coordinates = []
    h, w = image.shape[0:2]
    for i in range(0, detections.shape[2]):
      confidence = detections[0, 0, i, 2]
      if confidence > confidence_level:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")
        if startX > pad:
          startX = startX - pad
        if startY > pad:
          startY = startY - pad
        if endX + pad < w:
          endX = endX + pad
        if endY + pad < h:
          endY = endY + pad
        coordinates = [startX, startY, endX, endY]
        face = image[startY:endY, startX:endX]
        if face.size != 0:
          faces.append(face)

        # cv2.imwrite('face_{}.jpg'.format(i), face)
    # if startY - 10 > 10:
    #   y = startY - 10
    # else:
    #   y = startY + 10
    # cv2.rectangle(image, (startX - 20, startY - 20), (endX + 20, endY + 20),
    #               (0, 0, 255), 2)
    # cv2.putText(image, text, (startX, y),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    # cv2.imwrite('face_detected.jpg', image)
    return faces, coordinates

  def run_face_detector(self, image):
    self.img = image.copy()
    detections = self.detect_detections(self.img, self.prototxt_path,
                                        self.caffee_model_path)
    faces, coordiantes = self.filter_detections(self.img, self.confidence_level,
                                                detections)
    if len(faces) != 0:
      # print(
      #     "faces {}are detected for this {} confidence level".format(faces,
      #                                                                self.confidence_level))
      return faces[0], coordiantes, 0
    else:
      # Image is rotated clockwise 90 degree and detect face
      self.img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
      detections = self.detect_detections(self.img, self.prototxt_path,
                                          self.caffee_model_path)
      faces, coordiantes = self.filter_detections(self.img,
                                                  self.confidence_level,
                                                  detections)
      if len(faces) != 0:
        # print(
        #     "faces are detected for this {} confidence level with 90 degree rotation".format(
        #         self.confidence_level))
        return faces[0], coordiantes, 90
      else:
        # Image is rotated 180
        self.img = cv2.rotate(image, cv2.ROTATE_180)
        detections = self.detect_detections(self.img, self.prototxt_path,
                                            self.caffee_model_path)
        faces, coordiantes = self.filter_detections(self.img,
                                                    self.confidence_level,
                                                    detections)
        if len(faces) != 0:
          # print(
          # "faces are detected for this {} confidence level with 180 degree rotation".format(
          #     self.confidence_level))
          return faces[0], coordiantes, 180
        else:
          # rotated by 270 degree
          self.img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
          detections = self.detect_detections(self.img, self.prototxt_path,
                                              self.caffee_model_path)
          faces, coordiantes = self.filter_detections(self.img,
                                                      self.confidence_level,
                                                      detections)
          if len(faces) != 0:
            # print(
            #     "faces are detected for this {} confidence level with 270 degree rotation".format(
            #         self.confidence_level))
            return faces[0], coordiantes, 270
          else:
            # print("Could not find any faces in the image Bad quality image")
            return None, None, None

  def run(self, image=None, image_path=None):
    if image_path is not None:
      img = cv2.imread(image_path)
    else:
      img = image
    preprocessed_img = self.clahe_glare_removal(bgr_img=img)
    face, coordinates, degree = self.run_face_detector(preprocessed_img)
    return face, coordinates, degree

  def clahe_glare_removal(self, bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
