# Tesseract Quality Checker API

![project] ![research]



- <b>Project Mentor</b>
    1. Dr Uthayasanker Thayasivam
- <b>Contributors</b>
    1. Sabesan Sathananthan
    2. Sivaram Rasathurai
    3. Thineshan Panchalingam

---

## Summary
This API is used to check quality of the image which is send as an input for tesseract OCR. Tesseract OCR works best when there is a clear segmentation of the foreground text from the background. It is easy to instruct the user to get a best-fit image rather than doing preprocessing on the images.

# Description

## Constraints for the best-fit images
 - Appropriate width and height with DPI > 300 (For Driving License Appropriate width is 1000px and the height is 700px.
)
 - No Blur
 - No Glare
 - No shadow
 
 ## API Features
  - Resolution detection
  - Blur detection
  - Glare detection
  - Shadow/ Uneven illumination detection
  - Skew detection
  
  
## Quality Checker API Error Responses

|Status | Error Code | Error Messaging | Additional Info                                                   |
|-------|------------|-----------------|----------------|
| 400   | 10000      | DL is not detected   |Could not find driving license/NIC in uploaded image|
| 400   | 10001      | Skew detected   | Take the image without skew |
| 400   | 10002      | Resolution is not enough   | Atleast 1000px, 700px image is needed with 300 DPI |
| 400   | 10003      | Blur detected   |Check whether your image is properly focussed and keep your hands steady while capturing.|
| 400   | 10004     | Glare detected  | Try to avoid direct light  (flash)|
| 400   | 10005      | Illumination(shadow) detected | Need a good lighting environment. Take the image towards the light rather than away from the light.| 





