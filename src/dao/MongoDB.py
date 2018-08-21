from pymongo import MongoClient


class MongoDB(object):
    def __init__(self, settings):
        try:
            self.conn = MongoClient(settings["ip"], settings["port"])
        except Exception as e:
            print(e)
        self.db = self.conn[settings["db_name"]]
        self.my_set = self.db[settings["set_name"]]

    def insert(self, dic):
        self.my_set.insert(dic)

    def update(self, dic, newdic):
        self.my_set.update(dic, newdic)

    def delete(self, dic):
        self.my_set.remove(dic)

    def dbfind(self, dic):
        return self.my_set.find(dic)
