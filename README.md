# Face Detection & Tracking & Extract

   该工程可以多目标的人脸进行检测、跟踪并且提取人脸最优解(排除侧脸，选取正面)。
   
## Introduction
* **Dependencies:**
	* Python 3.5+
	* Dlib
	* Scikit-learn
	* MTCNN
	* Numpy
	* Scikit-image

## Run
* To run the python version of the code you have to put all the input videos in one folder like **/home/admin/videos** and then provide the path of that folder as command line argument:
```sh
python3 start.py /home/admin/videos 
```
* Then you can find extracted faces stored in the floder **./facepics** .


## Special Thanks to:
*  [**experimenting-with-sort**](https://github.com/ZidanMusk/experimenting-with-sort) 

