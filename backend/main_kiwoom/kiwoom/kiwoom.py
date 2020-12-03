import os
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from config.errorCode import *
from PyQt5.QtTest import *
from config.KiwoomType import *

class Kiwoom(QAxWidget):
    def __init__(self):

        super().__init__()
        print("kiwoom class")

        self.realType = RealType()
        ############변수 모음 #############
        self.account_num = None ##계좌번호
        self.deopsit = None ##예수금
        self.total_buy_money = None ## 총매입액
        self.total_profit_loss_rate = None ##총수익률(%)
        self.use_money=0
        self.use_money_percent =0.5

        self.account_stock_dict = {} ###보유종목 딕션너리.

        self.calcul_data =[]

        self.portfolio_stock_dict ={}

        ############################

        ######스크린 번호 모음 ####
        self.screen_my_info = "2000"
        self.screen_calculation_stock = "4000"
        self.screen_real_stock = "5000" ##종목별로 할당할 스크린 번호
        self.screen_meme_stock = "6000" ###종목 매매할 스크린 번호,.
        self.screen_start_stop_real = "1000"




        ######### Tr 요청 event loop 모음########
        self.login_event_loop = None ##얘는 로그인 이벤트 루프 tr과 조금 다름


        self.detail_account_info_event_loop = None
        self.account_balance_event_loop = None
        self.calculator_event_loop = QEventLoop()
        ###################################




        ######## Tr 요청 모음(메인 실행 코드)########

        ##로그인
        self.get_ocx_instance() ## ocx방식 제어 이벤트
        self.event_slots() ## 로그인 상태 이벤트
        self.signal_login_commConenect() ##로그인 요청.


        ##Tr
        self.get_account_info() ## 계좌번호 가져오기.
        self.detail_account_info() ## 예수금 가져오기.
        self.account_balance() ##계좌평가잔고 가져오기.



        self.read_code() #저장된 종목들 불러온다.

        self.screen_number_setting() # 스크린 번호 할당

        ##장 시작이냐 끝이냐 장중이냐
        self.dynamicCall("SetRealReg(Qstring, Qstring, Qstring, Qstring", self.screen_start_stop_real,'',self.real )




### PtQt5의 setControl class로 ocx방식 제어가능 ! (KHOPENAPI.KHOpenAPICtrl.1 는 openAPI 레지스터리임)
    def get_ocx_instance(self):
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")





#########이벤트 슬롯 걸어 놓기 ###########
    def event_slots(self):
        self.OnEventConnect.connect(self.login_slot) ###로그인 이벤트 걸어놓기.
        self.OnReceiveTrData.connect(self.trdata_slot)###TR요청 이벤트 걸어놓기 .

    def real_event_slots(self):
        self.OnReceiveRealData.connect(self.realdata_slot)





##########자동로그인 메소드 및 에러코드 시작 ######
    def login_slot(self, errCode):

        print(errors(errCode))

        self.login_event_loop.exit() ##이벤트 루프 끊기 .



    def signal_login_commConenect(self):
        self.dynamicCall("CommConnect()") ## dynamicCall은 PyQt5에서 다른서버에 전송을 가능하게함!

        self.login_event_loop = QEventLoop()###이벤트 루프 활성화
        self.login_event_loop.exec()###이벤트루프 끝이 안나게



############# 자동로그인 메소드 및 에러코드 끝 ##############






####################tr 정보 요청 메소드 #########################

    ##계좌번호
    def get_account_info(self):
        account_list = self.dynamicCall("GetLoginInfo(String)", "ACCNO")

        self.account_num = account_list.split(';')[0]
        print("나의 계좌 번호 : %s" % self.account_num) ##8149471511


    ##계좌정보
    def detail_account_info(self): ##예수금 가져오기.
        print("예수금 요청 ")
        ###예수금 요청시 필요 정보입력. ##
        self.dynamicCall("SetInputValue(String,String)", "계좌번호", self.account_num)
        self.dynamicCall("SetInputValue(String,String)", "비밀번호", "0000")
        self.dynamicCall("SetInputValue(String,String)", "비밀번호입력매체구분", "00")
        self.dynamicCall("SetInputValue(String,String)", "조회구분", "2")
        ###예수금 요청 ###
        self.dynamicCall("CommRqData(String,String,int,String)", "예수금상세현황요청", "opw00001", "0", "2000")

        ##요청뒤에는 무조건 이벤트 루프
        self.detail_account_info_event_loop = QEventLoop()
        self.detail_account_info_event_loop.exec_()

    ##계좌평가잔고내역
    def account_balance(self, sPrevNext = "0"):
        print("계좌평가잔고내역 요청")

        self.dynamicCall("SetInputValue(String,String)", "계좌번호", self.account_num)
        self.dynamicCall("SetInputValue(String,String)", "비밀번호", "0000")
        self.dynamicCall("SetInputValue(String,String)", "비밀번호입력매체구분", "00")
        self.dynamicCall("SetInputValue(String,String)", "조회구분", "2")

        self.dynamicCall("CommRqData(String,String,int,String)", "계좌평가잔고내역요청", "opw00018", sPrevNext, "2000")
        self.account_balance_event_loop = QEventLoop()
        self.account_balance_event_loop.exec_()

    ###종목 코드 반환
    def get_code_list_by_market(self, market_code):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)",market_code)
        code_list = code_list.split(";")[:-1]

        return code_list

    ###종목 분석 실험용
    def calculator_fnc(self):

        code_list = self.get_code_list_by_market('10')
        print("코스닥 갯수 %s" % len(code_list))

        for idx, code in enumerate(code_list):

            ###요청하기전에 스크린코드 끊기.
            self.dynamicCall("DisconnectRealData(QString)", self.screen_calculation_stock)

            print("%s / %s : 코스닥 코드 : %s 없데이트" % (idx+1, len(code_list), code))

            self.day_kiwoom_db(code=code)



    ### 종목코드로 주식 정보 받아오기.
    def day_kiwoom_db(self, code=None, data=None, sPrevNext ="0"):

        ###이벤트 멈추지 않고 딜레이를 줌.
        QTest.qWait(3600)


        self.dynamicCall("SetInputValue(QString, QString)", "종목코드", code)
        self.dynamicCall("SetInputValue(QString, QString)", "수정주가구분", "1")

        if data != None:
            self.dynamicCall("SetInputValue(QString, QString)", "기준일자", data)
        self.dynamicCall("CommRqData(QString, QString, int, QString)", "주식일봉차트조회", "opt10081", sPrevNext, self.screen_calculation_stock)


        self.calculator_event_loop.exec_()

    def read_code(self):
        if os.path.exists("files/condition_stock.txt"):
            f = open("files/condition_stock.txt", "r", encoding="utf8")

            lines = f.readlines()

            for line in lines:
                if line !="":
                    ls = line.split("\t") #["종목코드", "종목명","현재가"]

                    stock_code = ls[0]
                    stock_name = ls[1]
                    stock_price = int(ls[2].split("/n")[0])
                    stock_price = abs(stock_price)

                    ###{"2090923":{종목명:"삼성", 현재가:"60201"}, 203104:{종목명~
                    self.portfolio_stock_dict.update({stock_code : {"종목명" : stock_name, "현재가" : stock_price}})
            f.close()

            print(self.portfolio_stock_dict)

    def screen_number_setting(self):

        screen_overwrite = []

        #계좌평가잔고내역에 있는 종목들

        for code in self.account_stock_dict.keys():
            if code not in screen_overwrite:
                screen_overwrite.append(code)

        ## 포트폴리오에 담겨있는 종목들
        for code in self.portfolio_stock_dict.keys():
            if code not in screen_overwrite:
                screen_overwrite.append(code)

        #스크린번호 할당
        cnt = 0
        for code in screen_overwrite:

            temp_screen = int(self.screen_real_stock)
            meme_screen = int(self.screen_meme_stock)

            #스크린 하나당 종목 50개
            if (cnt % 50 ) == 0:
                temp_screen += 1
                self.screen_real_stock = str(temp_screen)

            if(cnt % 50 ) == 0:
                meme_screen += 1
                self.screen_meme_stock= str(meme_screen)

            if code in self.portfolio_stock_dict.keys():
                self.portfolio_stock_dict[code].update({"스크린번호": str(self.screen_real_stock)})
                self.portfolio_stock_dict[code].update({"주문용스크린번호": str(self.screen_meme_stock)})

            elif code not in self.portfolio_stock_dict.keys():
                self.portfolio_stock_dict.update({code: {"스크린번호": str(self.screen_real_stock), "주문문용스크린번호": str(self.screen_meme_stock) }})


            cnt += 1
        print(self.portfolio_stock_dict)

    def realdata_slot(self, sCode, sRealType, sRealData):
        print(sCode)























######### tr요청해서 이벤트에 걸려있는 정보를 가져요기.########################



    ####### 이 메소드로만 이벤트에 걸려있는 데이터 받아올꺼임. tr요청 받기 데이터 !####
    def trdata_slot(self, sScrNo, sRQName, sTrCode, sRecordName, sPrevNext):
        '''
        Tr요청을 받는 메소드 !.
        :param sScrNo: 스크린번호
        :param sRQName: 내가요청했을때 지은 이름
        :param sTrCode:  요청id, tr코드
        :param sRecordName: 사용안함
        :param sPrevNext: 다음페이지가 잇는지.
        :return:
        '''
        if sRQName == "예수금상세현황요청":
            deposit = self.dynamicCall("GetCommData(String, String, int, Sting)", sTrCode, sRQName, 0, "예수금")
            print("예수금 %s" % int(deposit))
            self.deopsit = deposit

            possible_deposit = self.dynamicCall("GetCommData(String,String,int,Sting)", sTrCode, sRQName, 0, "출금가능금액")
            print("출금가능금액 %s" % int(possible_deposit))

            self.use_money = int(possible_deposit)*self.use_money_percent
            self.use_money = self.use_money/4

            ###요청한 tr을 받으면 이벤트루프 끊어주기 ! 스텍큐 같은 개념.
            self.detail_account_info_event_loop.exit()


        if sRQName == "계좌평가잔고내역요청":
            total_buy_money = self.dynamicCall("GetCommData(String, String, int, Sting)", sTrCode, sRQName, 0, "총매입금액")
            print("총매입금액: %s" % int(total_buy_money))
            self.total_buy_money = total_buy_money

            total_profit_loss_rate = self.dynamicCall("GetCommData(String, String, int, Sting)", sTrCode, sRQName, 0, "총수익률(%)")
            print("총수익률(%%): %s %%" % float(total_profit_loss_rate))
            self.total_profit_loss_rate = total_profit_loss_rate
            self.account_balance_event_loop.exit()


            ###여러개 보유종목 가져오기.
            ### GetRepeatCnt는 최대 20개밖에 카운트롤 못해서 다음종목있을때 sPrevNext =2 로설정하고 넘어간담에 또받아야댐
            # GetRepeatCnt는는 멀티데이터 조회용. ! .
            rows = self.dynamicCall("GetRepeatCnt(QSting,QString)", sTrCode,sRQName)##종목 보유 개수
            cnt =0
            for i in range(rows):
                code_nm = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "종목명")

                code = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "종목번호")
                code = code.strip()[1:]

                stock_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "보유수량")
                buy_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매입가")
                learn_rate = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "수익률(%)")
                current_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "현재가")
                total_chegual_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매입금액")
                possible_quantity = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "매매가능수량")

                if code in self.account_stock_dict:
                    pass
                else:
                    self.account_stock_dict.update({code : {}})




                code_nm = code_nm.strip() ##공백지워주기.
                stock_quantity = int(stock_quantity.strip())
                buy_price = int(buy_price.strip())
                learn_rate = float(learn_rate.strip())
                current_price = int(current_price.strip())
                total_chegual_price = int(total_chegual_price.strip())
                possible_quantity = int(possible_quantity.strip())

                self.account_stock_dict[code].update({"종목명": code_nm})
                self.account_stock_dict[code].update({"보유수량" : stock_quantity})
                self.account_stock_dict[code].update({"매입가" : buy_price})
                self.account_stock_dict[code].update({"수익률(%)":learn_rate})
                self.account_stock_dict[code].update({"현재가" : current_price})
                self.account_stock_dict[code].update({"매입금액": total_chegual_price})
                self.account_stock_dict[code].update({"매매가능수량": possible_quantity})

                cnt += 1

            print("계좌에 가지고있는 종목 수  %s" % self.account_stock_dict)

            ###종목이 20개 넘어갈때 넘겨주고 이벤트루프 올려줘야댐.
            ###if sPrevNext == "2":
            ###    self.account_balance(sPrevNext="2")
            ###else:
            ###    self.account_balance_event_loop.exit()

        if "주식일봉차트조회" == sRQName:

            code = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, 0, "종목코드")
            code = code.strip()
            print("%s 일봉데이터 요청" % code)
            rows = self.dynamicCall("GetRepeatCnt(QSting,QString)", sTrCode, sRQName)
            print(rows)

            #한번 조회하면 600일치 받을수 있음.

            for i in range(rows): ## [0..599]
                data = []

                current_price = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "현재가")
                Volume = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "거래량")
                trading_Volume = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "거래대금")
                date = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "일자")
                Open = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "시가")
                High = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "고가")
                Low = self.dynamicCall("GetCommData(QString, QString, int, QString)", sTrCode, sRQName, i, "저가")

                data.append("")
                data.append(current_price.strip())
                data.append(Volume.strip())
                data.append(trading_Volume.strip())
                data.append(date.strip())
                data.append(Open.strip())
                data.append(High.strip())
                data.append(Low.strip())
                data.append("")

                self.calcul_data.append(data.copy())

            print(len(self.calcul_data))






            if sPrevNext == "2":
                self.day_kiwoom_db(code = code, sPrevNext = sPrevNext)

            else:

                ##조건의 부합하는 종목 뽑기(120일 이평선 그릴만큼 데이터가있는지)
                print("총 일수 %s", len(self.calcul_data))


                pass_success = False
                if self.calcul_data == None or len(self.calcul_data) <120:
                    total_price =0
                    for value in self.calcul_data[:120]: #오늘부터 ~199일전
                            total_price += int(value[1]) ##종가의 합

                    moving_average_price = (total_price / 120)

                    ###오늘자 주가가 120일 이편선에 걸쳐있는지 확인
                    bottom_stock_price = False
                    check_price = None

                    if int(self.calcul_data[0][7]) <= moving_average_price and moving_average_price <= int(self.calcul_data[0][6]):
                        print("오늘 주가 120이평선에 걸쳐잇는 것 확인")
                        bottom_stock_price = True
                        check_price = int(self.calcul_data[0][6])

                        ###과거 일봉들이 120일 이편선보다 밑에있는지
                        ## 그렇게 확인하다가 일봉이 120일 이평선보다 위에 있으면 계산 진행
                        prev_price = None ##과거의 일봉 저가
                    if bottom_stock_price == True:

                        moving_average_price = 0
                        price_top_moving = False

                        idx = 1
                        while True:
                            if len(self.calcul_data[idx:]):# 120일치가 있는지 계속 확인
                                print("120일치가 없음")
                                break
                            total_price=0
                            for value in self.calcul_data[idx:120+idx]:
                                total_price += int(value[1])
                            moving_average_price_prev = total_price / 120

                            if moving_average_price_prev <= int(self.calcul_data[idx][6]) and idx <=20:
                                print ("20일동안 주가가 120일 이평선과 같거나 위에있으면 조건 통과 못함")
                                price_top_moving = False
                                break
                            elif int(self.calcul_data[idx][7]) > moving_average_price_prev and idx >20 :
                                price_top_moving = True
                                prev_price = int(self.calcul_data[idx][7])
                                break

                            idx +=1

                        ## 해당부분 이평선이 가장 최근일자의 이편선가격보다 낮은지 확인
                        if price_top_moving == True:
                            if moving_average_price > moving_average_price_prev and check_price > prev_price:
                                print("포작된  이평선의 가격이 오늘자 이평선가격보다 낮은것이 확인")

                                pass_success = True


                if pass_success == True  :
                    print("조건부 통과됨")

                    code_nm = self.dynamicCall("GetMasterCodeName(Qstring)", code)

                    f = open("files/condition_stock.txt", "a", encoding="utf8")
                    f.write("%s\t%s\t%s\n" % code, code_nm,str(self.calcul_data[0][1]))
                    f.close()
                elif pass_success == False:
                    print("조건부 통과못함.")

                self.calcul_data.clear()

                #################조건부헙.
                #조건 쓰기.
                self.calculator_event_loop.exit()



















#



