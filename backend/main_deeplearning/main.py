

from tensorflow.keras.models import load_model
import tensorflow as tf

from DataProcessing.DataProcessing import *
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    code_dic = {'삼성전자': '005930', 'SK하이닉스': '000660', 'LG화학': '051910',
                    '삼성바이오로직스': '207940', '삼성전자우': '005935', '셀트리온': '068270',
                    'NAVER': '035420', '현대차': '005380', '삼성SDI': '006400',
                    '카카오': '035720', '기아차': '000270', 'LG생활건강': '051900',
                    '현대모비스': '012330', 'POSCO': '005490', '삼성물산': '028260'}


    probability= {}



    ##삼성전자 예측.

    now = datetime.datetime.now()
    end = (now.year, now.month, now.day)
    tf.compat.v1.disable_eager_execution()
    for i in code_dic.keys():

        nm = str('./Model/'+code_dic[i]+'.h5')
        model = load_model(nm)

        code = str(code_dic[i]+'.KS')
        stock = StockDataset(code, 60, 1, (2020, 1, 1), end)
        pred = stock.For_tommorow()
        y_pred_tommorow = model.predict(pred)
        up = y_pred_tommorow[0, 0]
        down = 1 - y_pred_tommorow[0, 0]
        probability[i] = {"up": str(up), "down": str(down)}
        print(probability)

    cred = credentials.Certificate('stock-b69a2-firebase-adminsdk-t55a0-66534d7508.json')
    # ##키 파일
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://stock-b69a2.firebaseio.com'})
    dir = db.reference('예측정보')
    # ## 데이터베이스 접근.
    #
    dir.update(probability)







# See PyCharm help at https://www.jetbrains.com/help/pycharm/
