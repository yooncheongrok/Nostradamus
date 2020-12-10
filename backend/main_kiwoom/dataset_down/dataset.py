import pandas as pd
from pykiwoom.kiwoom import *

code_dic = {'삼성전자': '005930', 'SK하이닉스': '000660', 'LG화학': '051910',
                    '삼성바이오로직스': '207940', '삼성전자우': '005935', '셀트리온': '068270',
                    'NAVER': '035420', '현대차': '005380', '삼성SDI': '006400',
                    '카카오': '035720', '기아차': '000270', 'LG생활건강': '051900',
                    '현대모비스': '012330', 'POSCO': '005490', '삼성물산': '028260'}

kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

dfs = []
df = kiwoom.block_request("opt10081",
                          종목코드="005935",
                          기준일자="20200424",
                          수정주가구분=1,
                          output="주식일봉차트조회",
                          next=0)

dfs.append(df)

while kiwoom.tr_remained:
    df = kiwoom.block_request("opt10081",
                              종목코드="005930",
                              기준일자="20200424",
                              수정주가구분=1,
                              output="주식일봉차트조회",
                              next=2)
    dfs.append(df)
    time.sleep(1)

df = pd.concat(dfs)

df_result = df.loc[:, ['시가', '고가', '저가', '종가', '거래량']][0:2500]
df_result.reset_index(inplace=True, drop=True)

df_result.to_csv("005935.csv")

# 2500개의 데이터(시가 고가 저가 종가(현재가) 거래량) 가져오기