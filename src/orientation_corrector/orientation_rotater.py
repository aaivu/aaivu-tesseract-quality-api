from orientation_corrector.face_detector import FaceDetector
from orientation_corrector.face_part_detector import run


def get_rotation_degree(image, prototxt_path, caffee_model_path,
    confidence_level, shape_predictor_path, fd=None):
  if fd is None:
    fd = FaceDetector(prototxt_path=prototxt_path,
                      caffee_model_path=caffee_model_path,
                      confidence_level=confidence_level)
  face, coordinates, first_degree = fd.run(image)
  print(first_degree)
  second_degree = None
  if face is not None:
    second_degree = run(face, shape_predictor_path=shape_predictor_path)
  print(second_degree)
  if first_degree is None:
    return second_degree
  if second_degree is None:
    return first_degree
  return first_degree + second_degree
