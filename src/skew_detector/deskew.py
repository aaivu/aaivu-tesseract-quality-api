""" Deskews file after getting skew angle """
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skew_detector.skew_detect import SkewDetect
from skimage.transform import rotate


class Deskew:

  def __init__(self, display_image, output_file, r_angle=0,
      input_img=None, input_file=None):
    self.input_img = input_img
    self.input_file = input_file
    self.display_image = display_image
    self.output_file = output_file
    self.r_angle = r_angle
    img = cv2.imread(input_file)
    if input_file:
      self.skew_obj = SkewDetect(input_file=input_file)
    else:
      self.skew_obj = SkewDetect(input_image=input_img)

  def deskew(self):
    if self.input_file:
      img = cv2.imread(self.input_file)
    else:
      img = self.input_img
    # img = io.imread(self.input_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = self.skew_obj.process_single_file()
    rot_angle = 0
    if 'Estimated Angle' in res.keys():
      angle = res['Estimated Angle']

      if angle >= 0 and angle <= 90:
        rot_angle = angle - 90 + self.r_angle
      if angle >= -45 and angle < 0:
        rot_angle = angle - 90 + self.r_angle
      if angle >= -90 and angle < -45:
        rot_angle = 90 + angle + self.r_angle

    rotated = rotate(img, rot_angle, resize=True)

    if self.display_image:
      self.display(rotated)

    if self.output_file:
      self.saveImage(rotated * 255)
    img = rotated * 255
    img = img.astype(np.uint8)
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    return img

  def saveImage(self, img):
    path = self.skew_obj.check_path(self.output_file)
    img = img.astype(np.uint8)
    # io.imsave(path, img)
    r, g, b = cv2.split(img)
    img = cv2.merge([b, g, r])
    cv2.imwrite(path, img)

  def display(self, img):

    plt.imshow(img)
    plt.show()

  def run(self):

    if (self.input_file or self.input_img is not None):
      return self.deskew()
