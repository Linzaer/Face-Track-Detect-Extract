# 人脸信息特征类
class Face():
    def __init__(self):
        self.__feature_vector = None
        self.__id = ''
        self.__ori_pic = ''
        self.__detect_pic = ''
        self.__scene_pic = ''

    @property
    def feature_vector(self):
        return self.__feature_vector

    @feature_vector.setter
    def feature_vector(self, feature_vector):
        self.__feature_vector = feature_vector

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def ori_pic(self):
        return self.__ori_pic

    @ori_pic.setter
    def ori_pic(self, ori_pic):
        self.__ori_pic = ori_pic

    @property
    def detect_pic(self):
        return self.__detect_pic

    @detect_pic.setter
    def detect_pic(self, detect_pic):
        self.__detect_pic = detect_pic

    @property
    def scene_pic(self):
        return self.__scene_pic

    @scene_pic.setter
    def scene_pic(self, scene_pic):
        self.__scene_pic = scene_pic
