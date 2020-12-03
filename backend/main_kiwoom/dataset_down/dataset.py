import pandas as pd
from pykiwoom.kiwoom import *

kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)
print("블록킹 로그인 완료")

dfs = []
df = kiwoom.block_request("opt10081",
                          종목코드="005930",
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

df_result = df.loc[:, ['시가', '고가', '저가', '현재가', '거래량']][0:2500]
df_result.reset_index(inplace=True, drop=True)

df_result.to_excel("005930.xlsx")

# 2500개의 데이터(시가 고가 저가 종가(현재가) 거래량) 가져오기