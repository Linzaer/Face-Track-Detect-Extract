# Face Detection & Tracking & Extract

   This project can **detect** , **track** and **extract** the **optimal** face in multi-target faces (exclude side face and select the optimal face).
   
## Introduction
* **Dependencies:**
	* Python 3.5+
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
* Then you can find extracted faces stored in the floder **./facepics** .

## Results
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic1.jpg "Logo Title Text 1")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic2.jpg "Logo Title Text 1")
![alt text](https://raw.githubusercontent.com/wiki/Linzaer/Face-Track-Detect-Extract/pic3.jpg "Logo Title Text 1")

## Special Thanks to:
*  [**experimenting-with-sort**](https://github.com/ZidanMusk/experimenting-with-sort) 

## License
MIT LICENSE

