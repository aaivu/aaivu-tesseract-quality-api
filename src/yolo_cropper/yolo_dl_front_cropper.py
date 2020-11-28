import cv2


def load_model(yolo_config='../assets/yolo_config.cfg',
    yolo_weight='../assets/yolo_weights.weights'):
  net = cv2.dnn_DetectionModel(yolo_config, yolo_weight)
  net.setInputSize(416, 416)
  net.setInputScale(1.0 / 256)
  return net


# net.SetInutSwapB(True)

def show_detection(net, img, padding=20):
  h, w, _ = img.shape
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  classes, confidences, coordinates = net.detect(img, confThreshold=0.6,
                                                 nmsThreshold=0.3)
  # img = cv2.resize(img, (416, 416))
  # cv2.imshow("resized img", img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  orig = img.copy()
  cv2.rectangle(orig, coordinates[0], (255, 0, 0), 2)
  print(coordinates)
  # cv2.imwrite("boxed_img.png", orig)
  # cv2.imshow('boxed', orig)
  if len(coordinates) == 1:
    [x, y, nw, nh] = coordinates[0]
    # print(y, x, nw, nh)
    if (nh + 2 * padding < h):
      nh = nh + 2 * padding
    else:
      nh = h
    if ((nw + 2 * padding) < w):
      nw = nw + 2 * padding
    else:
      nw = w
    if (x - padding > 0):
      x = x - padding
    else:
      x = 0
    if ((y - padding) > 0):
      y = y - padding
    else:
      y = 0
    print(x, y, nw, nh)

    return img[y:y + nh, x:x + nw]
  else:
    # print(f"Multiple objects are detected")
    return None


def run(image=None, image_path=None, yolo_config=None, yolo_weight=None,
    padding=10):
  if image is None:
    if image_path is None:
      # print("provide image or iamge path")
      return None
    else:
      image = cv2.imread(image_path)

  if (yolo_config is None and yolo_weight is None):
    model = load_model()
  else:
    model = load_model(yolo_config=yolo_config, yolo_weight=yolo_weight)
  yolo_img = show_detection(net=model, img=image, padding=padding)
  if yolo_img is not None:
    return 1, yolo_img
  else:
    return 10000, None
