from MongoDB import *

settings = {
    "ip": '192.168.18.61',  # ip
    "port": 27017,  # 端口
    "db_name": "face",  # 数据库名字
    # "set_name": "face"  # 集合名字
    # "set_name": "face_self"  # 集合名字
    # "set_name": "face_self_50w"  # 50w数据模型
    # "set_name": "face_main"  # 集合名字
    # "set_name": "face_no_margin"  # 集合名字
    # "set_name": "casia_mydata_90000_no_margin"  # 集合名字
    # "set_name": "50wdatamodel_no_margin"  # extract from 50w dataset model
    # "set_name": "50wdatamodel_32_margin"  # extract from 50w dataset model
    "set_name": "65wdatamodel_32_margin"  # extract from 50w dataset model
}

db = MongoDB(settings)


def insert_face(face):
    dict = {"feature_vector": face.feature_vector, "ori_pic": face.ori_pic, "detect_pic": face.detect_pic,
            "scene_pic": face.scene_pic}
    db.insert(dict)


def findall():
    dict = {"feature_vector": {'$size': 1}}
    return db.dbfind(dict)
