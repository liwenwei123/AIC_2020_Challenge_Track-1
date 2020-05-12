# AIC 2020 Challenge Track 1 by Orange-Control
<<<<<<< HEAD
This repository contains our code for Challenge Track 1: Vehicle Counts by Class at Multiple Intersections in AI City Challenge 2020.
=======
This repository contains the code by Team __Orange-Control__ for _Challenge Track 1: Vehicle Counts by Class at Multiple Intersections_ in AI City Challenge 2020.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd


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
<<<<<<< HEAD

=======
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd
```


### Inference

Here are the steps to reproduce our results:

<<<<<<< HEAD
1. Download the corresponding model file [best.pt](https://drive.google.com/open?id=1RC0weOuPemqMUuEUvi7nVTugaqyhlzaP) and put it in the folder `weights`
2. Make sure the raw video files and required txt files are in the folder `Data/dataset_A`
3. Run `inference.py` to get separate result files in the folder `output` for all 31 videos
4. Run `result.py` to combine all 31 csv files and get the single submission file `track1.txt`
=======
1. Download the corresponding model file [best.pt](https://drive.google.com/open?id=1BaCOU5ABwFMSjbc8frrAIpC6Dp0zTQJz) and put it in the folder `weights`.
2. Make sure the raw video files and required txt files are in the folder `Data/dataset_A`.
3. Run `inference.py` to get separate result files in the folder `output` for all 31 videos.
4. Run `result.py` to combine all 31 csv files and get the single submission file `track1.txt`.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd
```
mkdir weights
mkdir output
python3 inference.py 1 31
python3 result.py
```

<<<<<<< HEAD
We use YOLOv3+sort to detect and track vehicles. To count the movement, we use a detection line (detection line) for each movement by annotating the provided training videos (Data set A), as defined in `get_lines.py`. If a vehicle passes the detection line, the count of the corresponding movement will increase by 1 after a short pre-defined delay calculated based on the training data.


### Training the detector
We use yolov3 as our detector, which is initialized by the public COCO pre-trained model and fine-tuned with some annotated frames from the training set (which will be described later). The corresponding files are in the folder `yolov3_pytorch`.Some of the utility scripts are borrowed from https://github.com/ultralytics/yolov3. Below are the steps to train the detector.

1. Make sure the following files are placed correctly
	* The training images (extracted from the raw training videos) in `data/images/`
	* The annotation text files in `yolov3_pytorch/data/labels/`
	* `object.data` and `object.names` in `yolov3_pytorch/data`, which describe the input data and output classes
2. Downdload the official coco pretrained model [yolov3.weights](https://drive.google.com/open?id=1PfJ4nPGTF9OAIuN9IUkWwu4pRV6lPNxV) from [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) and put it in `yolov3_pytorch/weights`
=======
We use YOLOv3+sort to detect and track vehicles. To count the movement, we use an additional detection line for each movement by annotating the provided training videos (Data set A), as described in `get_lines.py`. If a vehicle passes the detection line, the count of the corresponding movement will increase by 1 after a short pre-defined delay calculated based on the training data.


### Training the detector
We use yolov3 as our detector, which is initialized by the public COCO pre-trained model and fine-tuned with some annotated frames from the training set (which will be described later). The corresponding files are in the folder `yolov3_pytorch`. Below are the steps to train the detector.

1. Make sure the following files are placed correctly.
	* The training images (extracted from the raw training videos) in `data/images/`
	* The annotation text files in `yolov3_pytorch/data/labels/`
	* `object.data` and `object.names` in `yolov3_pytorch/data`, which describe the input data and output classes
2. Downdload the official coco pretrained model [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) from [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/) and put it in `yolov3_pytorch/weights`.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd
3. Use the following train command to finetine the pretrained model. The `train_best.pt` file is the final model.
```
cd yolov3_pytorch
unzip data/labels.zip
python3 train.py --data data/object.data --cfg cfg/yolov3.cfg --epochs 200
```
<<<<<<< HEAD
=======
Some of the utility scripts are borrowed from https://github.com/ultralytics/yolov3.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd

##### Annotation Data

We selected 5 videos from the provided training videos (Data set A), including `cam3.mp4, cam5.mp4, cam7.mp4, cam8.mp4, cam20.mp4`. A subset of 3835 frames was extracted from these videos for manual annotation.

<<<<<<< HEAD
You can use the following command to extract frames directly from the videos.And put the frames under `yolov3_pytorch/data/images`
=======
You can use the following command to extract frames directly from the videos.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd
```
ffmpeg -i cam_x.mp4 -r 1 -f image2 yolov3_pytorch/data/images/%06d.jpg
```

<<<<<<< HEAD
=======
The extracted frames should be put in the folder `yolov3_pytorch/data/images`.
>>>>>>> 41aaac79f94c71de19e4a26f4bd6eaab03e7afdd
