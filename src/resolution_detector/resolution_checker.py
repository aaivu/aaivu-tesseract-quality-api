"""
Driving License Height is 53.98 mm and Width is 85.60 mm
300 ppi then the pixels should be
width = 1011 px
height = 665 px
We may have pads
"""
import cv2


def test_image_height_and_width(img, debug=False, fname='output.png',
    desired_height=600,
    desired_width=1000):
  h, w, _ = img.shape
  print(f"Height and width of the image are {h} {w}")

  if h >= desired_height and w >= desired_width:
    if debug:
      print(f"Very Perfect Image \nHeight - {h}\nWidth - {w}")
      # cv2.imwrite(f"classifier/perfect/{fname.split('.')[0]}_{h}_{w}.png", img)
    return True, 0

  elif h >= desired_height:
    if debug:
      print(f"Height - {h} is good...")
      cv2.imwrite(f"classifier/height/{fname.split('.')[0]}_{h}_{w}.png", img)
    return False

  elif w >= desired_width:
    if debug:
      print(f"Width - {w} is good....")
      cv2.imwrite(f"classifier/width/{fname.split('.')[0]}_{h}_{w}.png", img)
    return False

  else:
    if debug:
      print(f"Bad Image We got... \nHeight{h} and Width {w}")
      cv2.imwrite(f"classifier/bad/{fname.split('.')[0]}_{h}_{w}.png", img)
    return False


def run_resolution_filter(image=None, image_path=None, height=600, width=1000):
  """
  This will take the image which is correctly rotated yolo output. Initially, We
  are doing for driving licenses only, Will return 1 if the height and width
  are greater than 700 and 1100 pixels else it will return 10002

  :return:
  """
  result = False
  if image is not None:
    result = test_image_height_and_width(image, desired_width=width,
                                         desired_height=height)
  if image_path is not None and image is None:
    img = cv2.imread(image_path)
    result = test_image_height_and_width(img, desired_width=width,
                                         desired_height=height)

  if result:
    return 1
  else:
    return 10002

#
# if __name__ == '__main__':
#   from util.utility import get_images_as_dict
#
#   images = get_images_as_dict('../data/deskew/corrected')
#   for f, img in images.items():
#     test_image_height_and_width(img, f, perfectHeight=500, perfectWidth=1000)
