import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

'''
Novel Method to detect illumination 
 1. We assumed Cropper done it works perfectly
 2. Image is segmented to foreground and background 
 3. Perceived brightness channel histogram is analysed
 4. If More than one peak with the following conditions detected then we assumed
 It is illumination
  5. peaks>75 and peaks <250
'''


# method to get perceived brightness of the channel
def get_perceived_brightness(float_img):
  float_img = np.float64(float_img)  # unit8 will make overflow
  b, g, r = cv2.split(float_img)
  float_brightness = np.sqrt(
      (0.241 * (r ** 2)) + (0.691 * (g ** 2)) + (0.068 * (b ** 2)))
  brightness_channel = np.uint8(np.absolute(float_brightness))
  return brightness_channel


# Helper function to smooth histogram

def smooth_histogram(x, window_len=11, window='hanning'):
  if x.ndim != 1:
    raise ValueError("smooth only accepts 1 dimension arrays.")

  if x.size < window_len:
    raise ValueError("Input vector needs to be bigger than window size.")

  if window_len < 3:
    return x

  if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
    raise ValueError(
        "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

  s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

  if window == 'flat':  # moving average
    w = np.ones(window_len, 'd')
  else:
    w = eval('np.' + window + '(window_len)')

  y = np.convolve(w / w.sum(), s, mode='valid')
  return y


# Helper function to plot the histogram
def plot_histogram(histogram, col, Tb, output='output', debug=False):
  plt.plot(histogram, color=col)
  plt.ylabel("Count", c='b')
  plt.xlim([0, 256])
  plt.xlabel("Pixel range", c='b')
  plt.axvline(x=Tb, color='r', lw=2, ls='--',
              label=f'Average Brightness - {int(round(Tb))}')
  plt.legend()
  if debug:
    plt.savefig("hist.png", format="png")
  plt.show()


# Seperate forground and background using dilation and smoothing
def seperate_foreground_background(img, debug=False, output='output'):
  rgb_planes = cv2.split(img)
  background_planes = []
  foreground_planes = []
  normalized_foreground_planes = []
  normalized_bk_planes = []
  for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((11, 11), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 31)
    bk_norm_img = cv2.normalize(bg_img, None, alpha=0, beta=255,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    foreground_planes.append(diff_img)
    normalized_foreground_planes.append(norm_img)
    background_planes.append(bg_img)
    normalized_bk_planes.append(bk_norm_img)

  # foreground = cv2.merge(foreground_planes)
  normalized_foreground = cv2.merge(normalized_foreground_planes)
  normalized_bk = cv2.merge(normalized_bk_planes)
  background = cv2.merge(background_planes)

  return normalized_foreground, normalized_bk


# Get the peaks from the image

def local_find_peaks(background_img, smooth_windows_length=15, debug=False,
    output='output'):
  height = background_img.shape[0]
  width = background_img.shape[1]

  perceived_brightness = get_perceived_brightness(background_img)
  Tb = np.average(perceived_brightness)
  hist, bins = np.histogram(perceived_brightness.ravel(), 256, [0, 256])
  smoothed_hist = smooth_histogram(hist, smooth_windows_length)
  if debug:
    cv2.imwrite(f'frames/{output.split(".")[0]}_brightness_channel.png',
                perceived_brightness)
    plot_histogram(hist, 'b', Tb=Tb,
                   output=f'frames/{output.split(".")[0]}_hist.png',
                   debug=debug)
    plot_histogram(smoothed_hist, 'c', Tb=Tb,
                   output=f'frames/{output.split(".")[0]}_smoothed_hist.png',
                   debug=debug)

  delay = int((smooth_windows_length - 1) / 2)
  mean_hist = int(height * width / (256))
  peaks, _ = find_peaks(smoothed_hist[delay:], prominence=mean_hist)
  return peaks, Tb


def detect_illumination_hist(img, k=0.2, low_peak_threshold_value=125,
    debug=False, output='output'):
  foreground, background = seperate_foreground_background(img, debug, output)

  peaks, Tb = local_find_peaks(background, smooth_windows_length=15,
                               debug=debug,
                               output=output)

  background_brightness_threshold = k * Tb  # Background image of the bk sometimes exist so we need to filter out
  if debug:
    fname = output.replace('.png', '')
    cv2.imwrite(f'frames/{fname}_orig.png', img)
    cv2.imwrite(f'frames/{fname}_foreground.png', foreground)
    cv2.imwrite(f'frames/{fname}_background.png', background)
    print(
        f"Selected Peaks are {(peaks)} Tb {Tb} Background Threshod {background_brightness_threshold}\n")

  if len(peaks) > 1:  # Probabally Illumination
    lowest_peak = np.min(peaks)
    high_threshold_val = min(Tb, low_peak_threshold_value)
    if float(
        lowest_peak) > high_threshold_val:  # If the lowest peak is high than average brightness then the peaks will not occur of shadows
      return 1, peaks, Tb

    elif (background_brightness_threshold < lowest_peak) and (
        lowest_peak < low_peak_threshold_value):
      return 10005, peaks, Tb
    else:
      # lowest peak is considered as background of the image
      return 1, peaks, Tb
  else:
    if debug:
      print(f"No peaks or only one peak {peaks} is detected")
    return 1, peaks, Tb


def run(image=None, image_path=None, output_path=None, pad_ratio=0.075,
    debug=False):
  img = None
  if image_path is not None:
    img = cv2.imread(image_path)
  if image is not None:
    img = image
  if img is None:
    print("No Image source is provided")
    return None
  '''
  Since we need to detect the illumination inside the image we remove some edge parts from the image'''
  h, w = img.shape[0:2]
  pad = int(min(h, w) * pad_ratio)
  cropped = img[pad:(h - pad), pad:(w - pad)]

  code, selected_peaks, Tb = detect_illumination_hist(cropped,
                                                      debug=debug,
                                                      output=output_path)
  if debug:
    print(f"Pad value {pad}........")
    if output_path is not None:
      fname = output_path.replace('.png', '')
      cv2.imwrite(f'frames/{fname}_Tb_{Tb}_peaks{selected_peaks}.png', img)

  return code

# from util import utility
#
# if __name__ == '__main__':
#   folder_path = '../yolo_dl_front_cropper/yolocut'
#   # folder_path = 'classifier/detected'
#   images_with_names = utility.get_images_as_dict(folder_path)
#   fnames = list(images_with_names.keys())[:1000]
#   for fname in fnames:
#     if '.png' in fname:
#       image = cv2.imread(f"{folder_path}/{fname}")
#       result = run_v2(image)
#       if result == 1:
#         cv2.imwrite(f'classifier/good/{fname}', image)
#       else:
#         cv2.imwrite(f'classifier/shadow/{fname}', image)
#
# if __name__ == "__main__":
#   fname = '../yolo_dl_front_cropper/yolo_cut/gowsy/fdc_driver_license_fr_19c20a95dd0e1aa5a68bc21748b3f91f.png'
#   image = cv2.imread(fname)
#   cv2.imwrite("Orignal.png", image)
#   result = run(image, debug=True, output_path="output.png")
#   print(result)
#   cv2.waitKey(0)
#   cv2.destroyAllWindows()
