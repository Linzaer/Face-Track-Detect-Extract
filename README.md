# Face Detection & Tracking & Extract

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

   This project can **detect** , **track** and **extract** the **optimal** face in multi-target faces (exclude side face and select the optimal face).
   
## Introduction
* **Dependencies:**
	* Python 3.5+
	* Tensorflow
	* [**MTCNN**](https://github.com/davidsandberg/facenet/tree/master/src/align)
	* Scikit-learn
	* Numpy
	* Numba
	* Opencv-python
	* Filterpy

## Run
* To run the python version of the code :
```sh
python3 start.py
```
* Then you can find  faces extracted stored in the floder **./facepics** .
* If you want to draw 5 face landmarks on the face extracted,you just add the argument **face_landmarks**
```sh
python3 start.py --face_landmarks
```
## What can this project do?

* You can run it to extract the optimal face for everyone from a lot of videos and use it as a training set for **CNN Training**.
* You can also send the extracted face to the backend for **Face Recognition**.



## Results
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic4.gif "scene 1")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic5.jpg "faces extracted")

## Special Thanks to:
*  [**experimenting-with-sort**](https://github.com/ZidanMusk/experimenting-with-sort) 

## License
MIT LICENSE

