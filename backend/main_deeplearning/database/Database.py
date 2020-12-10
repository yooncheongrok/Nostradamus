import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

class Database(object):
    ########### 데이터베이스에 올리기.
    def __init__(self, probability):
        cred = credentials.Certificate('stock-b69a2-firebase-adminsdk-t55a0-66534d7508.json')
        ##키 파일
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://stock-b69a2.firebaseio.com'})
        dir = db.reference()
        ## 데이터베이스 접근.

        dir.update(probability)
        # In[13]: