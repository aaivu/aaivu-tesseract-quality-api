from flask import Response


class Error(Response):
  def __init__(self, status, errorCode, error, additional_info):
    super().__init__(status=status)
    self.errorCode = errorCode
    self.error = error
    self.additional_info = additional_info
    response = {'code': errorCode, 'error': error,
                'additional_info': additional_info}
    self.response = response
