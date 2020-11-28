import cv2

from orientation_corrector import orientation_rotater
from skew_detector.deskew import Deskew
from skew_detector.skew_detect import SkewDetect


def oculars_deskew(img=None, img_file=None):
  if img_file is not None:
    image = cv2.imread("img_file")
  if img is not None:
    image = img
  h, w, _ = image.shape
  # img = cv2.resize(image, (300, 300), interpolation=cv2.INTER_CUBIC)
  deskew_obj = Deskew(input_img=img, input_file=img_file,
                      output_file='deskew.png',
                      display_image=False, r_angle=0)
  o_img = deskew_obj.run()
  # output = cv2.resize(o_img, (h, w), interpolation=cv2.INTER_CUBIC)
  skew_detector = SkewDetect(input_image=image)
  # skew_detector.determine_skew()

  return o_img


def get_occulars_skew_detected_angle(img=None, img_file=None):
  if img is not None:
    img = cv2.resize(img, (200, 300))
    sd = SkewDetect(img)

  if img_file is not None:
    sd = SkewDetect(input_file=img_file)

  res = sd.process_single_file()

  estimate_angle = res['Estimated Angle']
  if estimate_angle >= 0 and estimate_angle <= 90:
    rot_angle = estimate_angle - 90
  if estimate_angle >= -45 and estimate_angle < 0:
    rot_angle = estimate_angle - 90
  if estimate_angle >= -90 and estimate_angle < -45:
    rot_angle = 90 + estimate_angle
  rot_angle = rot_angle * -1

  return rot_angle


def run_angle_detector(img, prototxt_path, caffee_model_path,
    shape_predictor_path, confidence_level=0.5, skew_threshold=5):
  degree = orientation_rotater.get_rotation_degree(img,
                                                   prototxt_path=prototxt_path,
                                                   caffee_model_path=caffee_model_path,
                                                   confidence_level=confidence_level,
                                                   shape_predictor_path=shape_predictor_path)
  rotated = img

  des = oculars_deskew(rotated)

  print(f"Degree -> {degree} rotated degree")
  rot_angle = get_occulars_skew_detected_angle(img)
  if degree is not None:
    if degree == 90:
      rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degree == 180:
      rotated = cv2.rotate(img, cv2.cv2.ROTATE_180)
    if degree == 270:
      rotated = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
  angles = [abs(rot_angle), abs(90 - rot_angle), abs(180 - rot_angle),
            abs(270 - rot_angle), abs(360 - rot_angle)]
  angle = min(angles)
  print(f"Angles List {angles}")
  if angle < skew_threshold:
    return 1, rotated
  else:
    return 10001, rotated


def skew_detector_run(img, prototxt_path, caffee_model_path,
    shape_predictor_path, confidence_level=0.8):
  return run_angle_detector(img, prototxt_path, caffee_model_path,
                            shape_predictor_path, confidence_level)


def occulars_full_deskew(img):
  deskewed = oculars_deskew(img)
  degree = orientation_rotater.get_rotation_degree(deskewed,
                                                   prototxt_path=prototxt_path,
                                                   caffee_model_path=caffee_model_path,
                                                   confidence_level=confidence_level,
                                                   shape_predictor_path=shape_predictor_path)
  if degree == 0:
    return deskewed

  if degree == 90:
    rotated = cv2.rotate(deskewed, cv2.ROTATE_90_CLOCKWISE)
    return rotated
  if degree == 180:
    rotated = cv2.rotate(deskewed, cv2.cv2.ROTATE_180)
    return rotated

  if degree == 270:
    rotated = cv2.rotate(deskewed, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated
