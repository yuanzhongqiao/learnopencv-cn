
# Post Training Quantization with OpenVino Toolkit

**This repository contains code for [Post Training Quantization with OpenVino Toolkit](https://learnopencv.com/post-training-quantization-with-openvino-toolkit/) blogpost**.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/xe8k0xbfdudomdy/AAAYS9r9c0jW7ypsFFSCNBPJa?dl=1)

This repository has the following as well,

* Python file to create the results JSON file for the COCO validation dataset.
* Juptyter notebook for calculating the mAP.

## Instructions

### Download the Video Used in the Post

* Download the video used in the post for inference from [this link](https://www.pexels.com/video/people-wearing-face-mask-in-public-area-4562551/).

### Getting the JSON Results File 

* To get the results JSON file for COCO validation set:

  * Execute `object_detection_demo_coco.py` by providing the correct path to the MS COCO validation dataset by editing the Python file.

  * Execute using the following commands:

    ```
    python object_detection_demo_coco.py --model tiny_yolov4_fp32/frozen_darknet_tiny_yolov4_model.xml -at yolo -i mscoco/val2017 --loop -t 0.2 --no_show -r -nireq 4
    ```

    ```
    python object_detection_demo_coco.py --model int8/optimized/yolo-v4-tiny.xml -at yolo -i mscoco/val2017 --loop -t 0.2 --no_show -r -nireq 4
    ```

### mAP Calculation

* Put the `pycocoEvalDemo.ipynb` in the `cocoapi/PythonAPI`.

* Run the `pycocoEvalDemo.ipynb` Notebook by providing the correct path the `results.json` 
* Once for the FP32 results.
* And once for INT8 results.
* The correct path to the MS COCO evaluation JSON file also needs to be provided. Please check the path according to your directory structure of the MS COCO dataset.


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/).