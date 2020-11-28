import cv2
import numpy as np

'''
This is a light weight glare detector which is used to detect the disability glare
in the images. 

Preprocessing step - Adjust contrast based on perceived brightness
Change colour space BGR to HSV

If S(x) < T(s) and V(x) > Tv then the pixel is detect as glare where T(s) is fixed
and T(v) is dynamic.

'''


class GlareDetector:
  def __init__(self, k, Ts, minimum_text_area=10.0,
      minimum_background_area=1000.0, ratio=0.3):
    self.fname = ""
    self.Ts = Ts
    self.k = k
    self.img = None
    self.height, self.width = None, None
    self.hsv_img = None
    self.detect_glare = False
    self.mask = None
    self.glare_places = None
    self.minimum_distance = None
    self.minimum_text_area_glare = minimum_text_area
    self.minimum_background_area_glare = minimum_background_area
    self.ratio = ratio

  # Store height, weight and hsv space image
  def set_img_params(self, img):
    self.img = img
    self.height, self.width, _ = img.shape
    self.hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    self.minimum_distance = min(self.height, self.width) * self.ratio

  def get_perceive_brightness(self, float_img):
    float_img = np.float64(float_img)  # unit8 will make overflow
    b, g, r = cv2.split(float_img)
    float_brightness = np.sqrt(
        (0.241 * (r ** 2)) + (0.691 * (g ** 2)) + (0.068 * (b ** 2)))
    brightness_channel = np.uint8(np.absolute(float_brightness))
    return brightness_channel

  '''
  Adjust the image contrast based on average perceive brightness of the image
  Default value is set to 80
  This preprocessing step will increase the glare detection more than histogram
  equalization.
  '''

  def contrast_equalization(self, average_brightness_threshold=80):
    contrast = 1
    img = self.img.copy()
    brightness_channel = self.get_perceive_brightness(img)
    avg_brightness = np.average(brightness_channel)

    # print("Initial Average brightness of the image {} ".format(avg_brightness))

    while avg_brightness > average_brightness_threshold:
      contrast = contrast - 0.01
      img = img * contrast
      temp_brightness = self.get_perceive_brightness(img)
      avg_brightness = np.average(temp_brightness)

    else:
      return img

  '''
  Method to get number of white pixels of the image
  '''

  def get_no_of_255_pixels(self):
    _, _, v = cv2.split(self.hsv_img)
    return np.count_nonzero(v == 255)

  '''
  Method to get the Ts and Tv values for the image
  '''

  def get_parameters(self):

    contrast_img = self.contrast_equalization(125)
    # cv2.imwrite(f"{self.fname}_contrast_adjusted.png", contrast_img)
    brightness_channel = self.get_perceive_brightness(contrast_img)
    Tb = np.average(brightness_channel)
    # print(Tb)
    size = self.height * self.width

    no_255_pixels = self.get_no_of_255_pixels()
    if no_255_pixels > size / 3:
      # print("The image has more than 1/3 of over brightness or white... ")
      # The thresholding step produces robust results un-
      # #der difficult conditions and allows a better control on
      # #the context
      # These values are set globally
      Tv = 245
      Ts = 30
    else:
      Tv = self.k * Tb
      Ts = self.Ts

    return Tb, Ts, Tv

  def get_glare_mask_region(self, Ts, Tv):
    lower_hsv = (0, 0, Tv)
    higher_hsv = (255, Ts, 255)
    # print(lower_hsv)
    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(self.hsv_img, lower_hsv, higher_hsv)
    mask = cv2.dilate(mask, (5, 5), iterations=5)
    self.mask = mask
    # Apply the mask on the image to extract the original color
    # cv2.imwrite(f'{self.fname}_mask.png', mask)
    # cv2.imshow('mask', mask)
    # print(mask.shape)
    no_255_pixels = np.count_nonzero(mask == 255)

    return mask, no_255_pixels

  def near_edge(self, img, cnt, ):
    """Check if a contour is near the edge in the given image."""
    x, y, w, h = cv2.boundingRect(cnt)

    # imgM = cv2.moments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # contorM = cv2.moments(cnt)
    # cx = int(imgM['m10']/imgM['m00'])
    # cy =int(imgM['m01']/imgM['m00'])
    # contx = int(contorM['m10']/contorM['m00'])
    # conty =int(contorM['m01']/ contorM['m00'])
    # print(cx)
    # distance_moments =  ((cx-contx)**2 +(cy-conty)**2)**0.5

    img_center = (self.height / 2, self.width / 2)
    cnt_center = (y + h / 2, x + w / 2)
    distance = ((img_center[0] - cnt_center[0]) ** 2 + (
        img_center[1] - cnt_center[1]) ** 2) ** 0.5
    print(
        f"cnt from center distance is {distance}, min Length{self.minimum_distance}", )
    if distance < self.minimum_distance:
      return True
    else:
      return False

  # Post processing step to remove very small white spots which has less area

  def get_filtered_contours(self, mask):
    # find contours
    cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    # self.draw_glare_places(contours=cntrs,k=0)

    img = self.img.copy()
    sorted_contours = sorted(cntrs, key=cv2.contourArea, reverse=True)
    filterd_contours = [cc for cc in sorted_contours if
                        (self.near_edge(img, cc))]
    big_area_contours = []
    for i in range(len(sorted_contours)):
      if cv2.contourArea(
          sorted_contours[i]) > self.minimum_background_area_glare:
        continue
      big_area_contours = sorted_contours[0:i]
      break

    if filterd_contours != []:
      print(len(filterd_contours))
      for k in range(len(filterd_contours)):
        if cv2.contourArea(filterd_contours[k]) < self.minimum_text_area_glare:
          filterd_contours = filterd_contours[0:k]
          break
    # outside_filtered_contours = [contour for contour in outside_contours if
    #                              cv2.contourArea(contour) > 100.0]
    total_contours = filterd_contours + big_area_contours

    return total_contours

  def draw_glare_places(self, contours):
    img = self.img.copy()
    for i in range(len(contours)):
      cv2.drawContours(img, contours, i, (0, 0, 255), thickness=3)
    # cv2.imwrite(f"{self.fname}_glare_places_{k}.png", img)
    # cv2.imshow("glare places", img)
    self.glare_places = img
    no_of_spots = len(contours)
    glare_size = cv2.contourArea(contours[0])

    return img, no_of_spots, glare_size

  '''
  Follwing methods are used to find the image is follwing the guassian or nor
  '''

  def run(self, img, fname=None):
    if fname is not None:
      self.fname = fname
    # print("Glare Detector is ready to run >>>>>>>>>>>>>>>>>>>>>")
    self.set_img_params(img)
    Tb, Ts, Tv = self.get_parameters()
    # print(
    #     "Parameters are derived \n Saturation Threshold {}\n Value Threshold {}".format(
    #         Ts, Tv))
    mask, area = self.get_glare_mask_region(Ts, Tv)
    print(" Area of glare in Image ", area)
    if area > 0:
      contours = self.get_filtered_contours(mask)
      if contours != []:
        # print("Glare is detected in image")
        glare_places, number_of_glares, glare_size = self.draw_glare_places(
            contours, k=1)
        return True, glare_places, number_of_glares, glare_size
      else:
        return False, None, 0, 0
    else:
      # print("No Glare in  full image")
      return False, None, 0, 0


def run(image=None, image_path=None, output=None, pad_ratio=0.075):
  img = None
  if image_path is not None:
    img = cv2.imread(image_path)
  if image is not None:
    img = image
  glare_detector = GlareDetector(k=2, Ts=40, minimum_text_area=10.0,
                                 minimum_background_area=1000.0, ratio=0.4)
  h, w = img.shape[0:2]
  pad = int(min(h, w) * pad_ratio)
  cropped = img[pad:(h - pad), pad:(w - pad)]
  # cv2.imshow("Cropped Image", cropped)
  if output is not None:
    is_glare, glare_places, number_of_glares, glare_size = glare_detector.run(
        cropped, fname=output)
  else:
    is_glare, glare_places, number_of_glares, glare_size = glare_detector.run(
        cropped)

  if is_glare:
    return 10004
  else:
    return 1


'''
The  Solution will be a lightweight glare detection that can be used in real-time.

For glare detection, we need to adjust the contrast as a preprocessing step.
For that, I calculate the perceived brightness using RGB channels. as below

```
 def get_perceive_brightness(float_img):
    float_img = np.float64(float_img)  # unit8 will make overflow
    b, g, r = cv2.split(float_img)
    float_brightness = np.sqrt(
        (0.241 * (r ** 2)) + (0.691 * (g ** 2)) + (0.068 * (b ** 2)))
    brightness_channel = np.uint8(np.absolute(float_brightness))
    return brightness_channel
```
With this brightness channel calculation, we calculate the average brightness of the image. If our brightness is higher than our threshold, then we do contrast adjustment to the image to make the image with an average brightness threshold value.

```
 def adjust_contrast( img, average_brightness_threshold=80):
    contrast = 1
    brightness_channel = get_perceive_brightness(img)
    avg_brightness = np.average(brightness_channel)

    print("Initial Average brightness of the image {} ".format(avg_brightness))

    while avg_brightness > average_brightness_threshold:
      contrast = contrast - 0.01
      img = img * contrast
      temp_brightness = get_perceive_brightness(img)
      avg_brightness = np.average(temp_brightness)

    else:
      return img
```

We take S and V channels of HSV to determine the glare of an image. We set the threshold value of a V channel as 2* average brightness above the threshold we assume that it might be a glare. Next, We consider the saturation channel to confirm the glare Here we are going to set a constant threshold if the saturation value is less than 30 then It might be glare. So we use both thresholds to determine glare as
below 
```
if Tv> 2*Tb and Ts<30 then
 image is detected as glare
```
So with our native python, we can do this

```
 image = cv2.imread("test.png")
 original = image.copy()
  # First we do the contrast adjustment
 adjusted_img = adjust_contrast(image, 125)
 brightness_channel = get_perceive_brightness(adjusted_img)

#Get the perceived brightness of the preprocessed image
 Tb = np.average(brightness_channel)

 Tv = 2*Tb
 Ts =30
 lower_hsv = (0, 0,Tv)
 higher_hsv = (255, Ts, 255)
 we changed the  image into HSV colour space
 hsv_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2HSV)
 # We find the glare region
 mask = cv2.inRange(hsv_img, lower_hsv, higher_hsv)
 mask = cv2.dilate(mask, (5, 5), iterations=5)
 mask_area = np.count_nonzero(mask == 255)
 if maks_area > 0:
  # find contours
  cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
  sorted_contours = sorted(cntrs,key=cv2.contourArea, reverse=True)
  #We filter out the contours which has less than 10 pixel area
  filterd_contours = []
  if sorted_contours != []:
   for k in range(len(sorted_contours)): 

    if cv2.contourArea(sorted_contours[k]) < 10.0:
     filterd_contours = sorted_contours[0:k]
      break
    cv2.drawContours(original, contours, i, (0, 0, 255), thickness=3)

   cv2.imwrite("glare_places.png", original)
  else:
   print("No glare is detected")

'''

# if __name__ == "__main__":
#   fname = '../yolo_dl_front_cropper/yolo_cut/dilax/fdc_driver_license_fr_19bd649691dfe9a25ef25ca4e4a9cee5.png'
#   image = cv2.imread(fname)
#   cv2.imwrite("Orignal.png", image)
#   result = run(image, output=fname.split('.')[0])
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()

# images = get_images_as_dict("../data/glareImages/", names_only=True)
# for fname in images:
#   filename = f"../data/glareImages/{fname}"
#   run(image_path=filename, output=fname)
#
# # run(
#     image_path="../data/glareImages/dc_driver_license_fr_2edbaceaedf5548807b13e93eefcefaa.png",
#     output="siva")
#
# if __name__ == "__main__":
#   from util import utility
#
#   folder_path = '../yolo_dl_front_cropper/yolocut'
#   # folder_path = 'classifier/detected'
#   images_with_names = utility.get_images_as_dict(folder_path)
#   fnames = list(images_with_names.keys())[:1000]
#   for fname in fnames:
#     if '.png' in fname:
#       image = cv2.imread(f"{folder_path}/{fname}")
#       result = run(image, output=fname.split('.')[0])
#       if result == 1:
#         cv2.imwrite(f'classifier/not_detected/{fname}', image)
#       else:
#         cv2.imwrite(f'classifier/detected/{fname}', image)
#   #
#   # img = cv2.imread(
#   #     "../data/yolo/dc_driver_license_fr_b20039688bd71001fca7cc5cb6ebdbba.png")
#   # result = run(img,
#   #              output='dc_driver_license_fr_4f9b5088b47374096b6057c074dcdcca')
#   print(result)
