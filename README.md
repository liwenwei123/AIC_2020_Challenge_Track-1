# AIC 2020 Challenge Track 1 by Orange-Control
This repository contains our code for Challenge Track 1: Vehicle Counts by Class at Multiple Intersections in AI City Challenge 2020.


### Requirements
* python3.6

```
torch>=1.5
torchvision==0.4
opencv-python==4.2
pandas==0.24.2
numpy==1.17.3
sklearn==0.21.2
filterpy==1.4.5
easydict==1.9
ffmpeg==2.8.15
numba==0.46.0
tensorboard>=1.14
```


### Inference

Here are the steps to reproduce our results:

1. Download the corresponding model file [best.pt](https://drive.google.com/open?id=1Usf1lidUOiUXchiaZHU9Q3HiLiBf9Npf) and put it in the folder `weights`
2. Make sure the raw video files and required txt files are in the folder `Data/dataset_A`
3. Run `track.py` to get separate result files in the folder `output` for all 31 videos
4. Run `result.py` to combine all 31 csv files and get the single submission file `track1.txt`
```
mkdir output
python track.py 1 31
python result.py
```

We use YOLOv3+sort to detect and track vehicles. To count the movement, we use a detection line (detection line) for each movement by annotating the provided training videos (Data set A), as defined in `get_lines.py`. If a vehicle passes the detection line, the count of the corresponding movement will increase by 1 after a short pre-defined delay calculated based on the training data.


### Training the detector
We use yolov3 as our detector, which is initialized by the public COCO pre-trained model and fine-tuned with some annotated frames from the training set (which will be described later). The corresponding files are in the folder `yolov3_pytorch`.Some of the utility scripts are borrowed from https://github.com/ultralytics/yolov3. Below are the steps to train the detector.

1. Make sure the following files are placed correctly
	* The training images (extracted from the raw training videos) in `data/images/`
	* The annotation text files in `yolov3_pytorch/data/labels/`
	* `object.data` and `object.names` in `yolov3_pytorch/data`, which describe the input data and output classes
2. Run `yolov3_pytorch/weights/download_yolov3_weights.sh` to downdload the official coco pretrained model from [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
3. Use the following train command to finetine the pretrained model. The `best.pt` file is the final model.
```
cd yolov3_pytorch
unzip data/labels.zip
./weights/download_yolov3_weights.sh
python3 train.py --data data/object.data --cfg cfg/yolov3.cfg --epochs 200
```

##### Annotation Data

We selected 5 videos from the provided training videos (Data set A), including `cam3.mp4, cam5.mp4, cam7.mp4, cam8.mp4, cam20.mp4`. A subset of 3835 frames was extracted from these videos for manual annotation.

You can use the following command to extract frames directly from the videos.
```
ffmpeg -i cam_x.mp4 -r 1 -f image2 yolov3_pytorch/data/images/%06d.jpg
```

