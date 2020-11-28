# import the necessary packages
import cv2


def variance_of_laplacian(image):
  return cv2.Laplacian(image, cv2.CV_64F).var()


def detect_blur(gray_img, threshold):
  fm = variance_of_laplacian(gray_img)
  # if the focus measure is less than the supplied threshold, then the image
  # should be considered "blurry".
  if fm < threshold:
    # print(f"{fm} amount Blur is detected")
    return 10003, fm
  else:
    # print(f"Blur is not in the image focus measure {fm}")
    return 1, fm


def run(image=None, image_path=None, debug=False,
    output_path="blur_tested.png",
    threshold=50):
  if image_path is not None:
    image = cv2.imread(image_path)
  if image is not None:
    image = image
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if gray.shape[0] > gray.shape[1]:
    gray = cv2.resize(gray, (400, 600))
  else:
    gray = cv2.resize(gray, (600, 400))
  code, fm = detect_blur(gray, threshold=threshold)
  if debug:
    fname = output_path.replace('.png', '')
    cv2.imwrite(f'{fname}_focus_measure_{fm}.png', image)
  return code

#
# if __name__ == "__main__":
#   img = cv2.imread(
#       "../data/images/fdc_driver_license_fr_0a2f0832388fd0cce66ab891a6395d41.png")
#   result = run(img)
#   print(result)
