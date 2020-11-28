import cv2
from blur_detector import blur_detector
from flask_api.error import Error
from glare_detector import glare_detector
from illumination_detector import illumination_detector
from resolution_detector import resolution_checker
from skewer import deskew_ocr
from yolo_dl_front_cropper import yolo_dl_front_cropper as yolo


def run(img=None, image_path=None, output_file=None, type="License Front",
    yolo_cut=False, height=700, width=1000):
  '''
  This method takes the raw image from the user run the yolo detector on the image.
  After yolo detection, Image will be deskewed and rotated correctly
  Height and width of the image will be checked.
  Blur detection
  Glare detection
  Illumination detection
  this will return the error any of quality missing and will return true if the image is perfect.
  :param width: Default width
  :param height: Height for the image default
  :param img:
  :param image_path:
  :param output_file:
  :return: error, Yolo cut Image
  '''
  if not yolo_cut:
    yolo_response, yolo_img = yolo.run(image=img, image_path=image_path,
                                       yolo_config="assets/yolo_config.cfg",
                                       yolo_weight='assets/yolo_weights.weights')
  # houged_img = hough_cropper.run_crop(yolo_img, output=output_file)
  # deskewed_img = oculars_deskew(houged_img)
  # cv2.imwrite("original.png", yolo_img)
  else:
    yolo_response = 1
    yolo_image = img

  if yolo_response == 1:
    # cv2.imwrite(f"data/yolo/{output_file}.png", yolo_img)

    # cv2.imwrite(f"data/hough/{output_file}", houged_img)
    # cv2.imwrite(f"data/deskew/{output_file}", deskewed_img)

    skew_response, rotated_img = deskew_ocr.run_angle_detector(yolo_img,
                                                               prototxt_path='assets/deploy.prototxt.txt',
                                                               caffee_model_path='assets/res10_300x300_ssd_iter_140000.caffemodel',
                                                               shape_predictor_path='assets/shape_predictor_68_face_landmarks.dat',
                                                               confidence_level=0.8,
                                                               skew_threshold=5)
    yolo_img = rotated_img
    # cv2.imwrite("rotated.png",rotated_img)
    # cv2.imshow("Rotated", rotated_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if skew_response == 1:
      # The image doesn't have skew....
      resolution_response = resolution_checker.run_resolution_filter(
          rotated_img,
          height=height,
          width=width)
      if resolution_response == 1:
        # The image has perfect height and width
        blur_response = blur_detector.run(image=rotated_img,
                                          image_path=image_path,
                                          threshold=80)
        if blur_response == 1:
          # Blur is not detected in Image
          glare_response = glare_detector.run(image=rotated_img,
                                              image_path=image_path)
          if glare_response == 1:
            # The glare is not detected in image
            illumination_response = illumination_detector.run(image=rotated_img,
                                                              image_path=image_path)
            if illumination_response == 1:
              # The illumination is  not detected in image
              return True, yolo_img
            else:
              error = Error(status=400, errorCode=illumination_response,
                            error="Uneven illumination Detected",
                            additional_info="Need a good lighting environment. "
                                            "Take the image towards the light"
                                            " rather than away from the light")
              return error, yolo_img
          else:
            error = Error(status=400, errorCode=glare_response,
                          error="Glare detected",
                          additional_info="Try to avoid direct light (flash)")
            return error, yolo_img
        else:
          error = Error(status=400, errorCode=blur_response,
                        error="Blur detected",
                        additional_info="Check whether your image is properly focussed "
                                        "& keep your hands steady while capturing.")
          return error, yolo_img

      else:
        error = Error(status=400, errorCode=resolution_response,
                      error="Resolution is not enough",
                      additional_info="Minimum 1000px, 600px image is "
                                      "needed with 300 DPI")
        return error, yolo_img
    else:
      error = Error(status=400, errorCode=skew_response,
                    error="Skew detected",
                    additional_info="Take the image without skew")
      return error, yolo_img


  else:
    error = Error(status=400, errorCode=yolo_response,
                  error=f"{type} is not detected",
                  additional_info=f"Capture and Upload the {type} image")
    return error, None


# run(image_path=folder_path, output_file="test")
from util import utility as util

if __name__ == '__main__':
  folder_path = 'data/images/resolution'
  images_with_names = util.get_images_as_dict(folder_path)
  #   util.remove_and_create_dir('data/noLicenseImages')
  #   util.remove_and_create_dir('data/skewImages')
  #   util.remove_and_create_dir('data/lowResolutionImages')
  #   util.remove_and_create_dir('data/blurImages')
  #   util.remove_and_create_dir('data/glareImages')
  #   util.remove_and_create_dir('data/illuminationImages')
  #   util.remove_and_create_dir('data/goodImages')
  #
  print(len(images_with_names.keys()))
  for fname, image in images_with_names.items():
    try:
      if str.lower(fname.split('.')[-1]) in ["png", "jpeg", "jpg"]:
        print(fname)
        image = cv2.imread(f"{folder_path}/{fname}")
        data = None
        result, yolo_out = run(image, output_file=fname, height=400, width=600)
        if type(result) == Error:
          print(f"Result of the image {result.error}")
          if result.errorCode == 10000:
            cv2.imwrite(f"data/yolo/{fname}", image)
          if result.errorCode == 10001:
            cv2.imwrite(f"data/images/skew/{fname}", yolo_out)
          if result.errorCode == 10002:
            cv2.imwrite(f"data/images/resolution/{fname}", yolo_out)
          if result.errorCode == 10003:
            cv2.imwrite(f"data/images/blur/{fname}", yolo_out)
          if result.errorCode == 10004:
            cv2.imwrite(f"data/images/glare/{fname}", yolo_out)
          if result.errorCode == 10005:
            cv2.imwrite(f"data/images/illumination/{fname}", yolo_out)
        else:
          cv2.imwrite(f"data/images/good/{fname}", yolo_out)
    except Exception as e:
      print(f"Exception {e} is occurred in {fname} ")

#           data = {"errorCode": result.errorCode,
#                   "error": result.error,
#                   "additionalInfo": result.additional_info}
#         print(data)
#
#         if type(result) == bool:
#           cv2.imwrite(f"data/goodImages/{fname}", yolo_out)
#
#         util.write_file(img_path=fname, text=data,
#                         output_folder='data/images')
#     except Exception as e:
#       print(f"{fname}  Error found {e}")
