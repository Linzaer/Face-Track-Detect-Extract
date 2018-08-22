# Face Detection & Tracking & Extract

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)
![Travis (.org)](https://img.shields.io/travis/USER/REPO.svg)

   This project can **detect** , **track** and **extract** the **optimal** face in multi-target faces (exclude side face and select the optimal face).
   
## Introduction
* **Dependencies:**
	* Python 3.5+
	* Tensorflow
	* [**MTCNN**](https://github.com/davidsandberg/facenet/tree/master/src/align)
	* Dlib
	* Scikit-learn
	* Numpy
	* Scikit-image

## Run
* To run the python version of the code you have to put all the input videos in one folder like **/home/admin/videos** and then provide the path of that folder as command line argument:
```sh
python3 start.py /home/admin/videos 
```
* Then you can find  faces extracted stored in the floder **./facepics** .
* If you want to draw 5 face landmarks on the face extracted,you can make the argument **face_landmarks** to be **True**


## Results
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic1.jpg "scene 1")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic2.jpg "scene 2")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic3.jpg "faces extracted")

## Special Thanks to:
*  [**experimenting-with-sort**](https://github.com/ZidanMusk/experimenting-with-sort) 

## License
MIT LICENSE

