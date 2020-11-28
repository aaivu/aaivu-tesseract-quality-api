# Project Title / Research Title

![project] ![research]



- <b>Project Lead(s) / Mentor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)
- <b>Contributor(s)</b>
    1. Name (talk forum profile link)
    2. Name (talk forum profile link)

<b>Usefull Links </b>

- GitHub : <project_url>
- Talk Forum : <talk_forum_link>

---

## Summary

two - three paragraphs about your project.

## Description

# Quality Checker API
This API is used to check quality of the image.This API will filter out the images which are not matched with following conditions

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






## More references

1. Reference
2. Link

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
