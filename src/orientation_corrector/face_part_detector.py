import cv2
import dlib
import imutils
import numpy as np
from imutils.face_utils.helpers import shape_to_np, FACIAL_LANDMARKS_IDXS

shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
def preprocess_img(image=None, image_path=None):
  if image_path is not None:
    img = cv2.imread(image_path)
  else:
    img = image
  img = imutils.resize(img, width=500)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return gray


def run(image=None, image_path=None,
    shape_predictor_path='shape_predictor_68_face_landmarks.dat',
    output='output'):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(shape_predictor_path)
  gray = preprocess_img(image=image, image_path=image_path)

  # detect faces in the grayscale image
  rects = detector(gray, 1)
  rotate_degree = 0
  while not rects:
    gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    rects = detector(gray, 1)
    rotate_degree = rotate_degree + 90
    if rotate_degree == 360:
      break

  # loop over the face detections
  for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = shape_to_np(shape)

    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    (nStart, nEnd) = FACIAL_LANDMARKS_IDXS['nose']
    (mStart, mEnd) = FACIAL_LANDMARKS_IDXS['mouth']
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]
    nosePts = shape[nStart:nEnd]
    mouthPts = shape[mStart:mEnd]
    # print("+++++++++++++++++++++++++")

    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    noseCenter = nosePts.mean(axis=0).astype('int')
    mouthCenter = mouthPts.mean(axis=0).astype('int')
    middle_eye_point = np.array([(leftEyeCenter[0] + rightEyeCenter[0]) / 2,
                                 (leftEyeCenter[1] + rightEyeCenter[1]) / 2],
                                dtype=np.int)
    dY = abs(rightEyeCenter[1] - leftEyeCenter[1])
    dX = abs(rightEyeCenter[0] - leftEyeCenter[0])
    angle_between_eyes = np.degrees(np.arctan2(dY, dX))
    print(middle_eye_point, noseCenter, mouthCenter, angle_between_eyes)
    if angle_between_eyes < 45:
      if noseCenter[1] > middle_eye_point[1]:
        # print(f"Already rotated degree {rotate_degree} No need to rotate")
        return rotate_degree
      else:
        print(f"Already rotated degree {rotate_degree} Need to rotate 180")
        return rotate_degree + 180
