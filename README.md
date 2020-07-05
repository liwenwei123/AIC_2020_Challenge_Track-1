# AIC 2020 Challenge Track 1 by Orange-Control

[3rd place](https://www.aicitychallenge.org/challenge-winners-2020/) (with **the best Efficiency Score**) in _Challenge Track 1: Multi-Class Multi-Movement Vehicle Counting_ in __2020 AI City Challenge__ @ CVPR. 

By Team __Orange-Control__ from <sup>1</sup>_AI Labs, Didi Chuxing (DiDi AI Labs)_ and <sup>2</sup>_Beijing University of Posts and Telecommunications (BUPT)_.
* Team Members: Wenwei Li<sup>2,1</sup>, Haowen Wang<sup>2,1</sup>, Yue Shi<sup>1</sup>, Ke Dong<sup>1</sup>, Bo Jiang<sup>1</sup>, Zhengping Che<sup>1</sup> _(team leader)_, Jian Tang<sup>1</sup> _(advisor)_, and Xiuquan Qiao<sup>2</sup> _(advisor)_


### Requirements
* python3.6

```
torch==1.5.0+cu101
torchvision==0.6.0+cu101
opencv-python==4.2
pandas==0.24.2
numpy==1.17.3
sklearn==0.21.2
filterpy==1.4.5
easydict==1.9
ffmpeg==2.8.15
numba==0.46.0
tensorboard>=1.14
pycocotools==2.0.0
tqdm==4.33.0
pillow==6.1.0

```


### Inference

Here are the steps to reproduce our results:

1. Download the corresponding model file [best.pt](https://drive.google.com/open?id=1BaCOU5ABwFMSjbc8frrAIpC6Dp0zTQJz) and put it in the folder `weights`.
2. Make sure the raw video files and required txt files are in the folder `data/Dataset_A`.
3. Run `inference.py` to get separate result files in the folder `output` for all 31 videos.
4. Run `result.py` to combine all 31 csv files and get the single submission file `track1.txt`.

```
mkdir weights
mkdir output
python3 inference.py 1 31
python3 result.py
```

We use YOLOv3+sort to detect and track vehicles. To count the movement, we use a detection line (detection line) for each movement by annotating the provided training videos (Data set A), as defined in `get_lines.py`. If a vehicle passes the detection line, the count of the corresponding movement will increase by 1 after a short pre-defined delay calculated based on the training data.

### Training the detector
We use yolov3 as our detector, which is initialized by the public COCO pre-trained model and fine-tuned with some annotated frames from the training set (which will be described later). The corresponding files are in the folder `yolov3_pytorch`. Below are the steps to train the detector.

1. Make sure the following files are placed correctly.
	* The training images (extracted from the raw training videos) in `data/images/`
	* The annotation text files in `yolov3_pytorch/data/labels/`
	* `object.data` and `object.names` in `yolov3_pytorch/data`, which describe the input data and output classes
2. Downdload the official coco pretrained model [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) from [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) and put it in `yolov3_pytorch/weights`.
3. Use the following train command to finetine the pretrained model. The `train_best.pt` file is the final model.
```
cd yolov3_pytorch
unzip data/labels.zip
python3 train.py --data data/object.data --cfg cfg/yolov3.cfg --epochs 200
```
Some of the utility scripts are borrowed from https://github.com/ultralytics/yolov3.

##### Annotation Data

We selected 5 videos from the provided training videos (Data set A), including `cam3.mp4, cam5.mp4, cam7.mp4, cam8.mp4, cam20.mp4`. A subset of 3835 frames was extracted from these videos for manual annotation.

You can use the following command to extract frames directly from the videos.
```
ffmpeg -i cam_x.mp4 -r 1 -f image2 yolov3_pytorch/data/images/%06d.jpg
```
The extracted frames should be put in the folder `yolov3_pytorch/data/images`.

