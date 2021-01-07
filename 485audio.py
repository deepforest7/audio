# coding=utf-8
import serial
import time
import csv
import requests
import configparser
import os

cfg_file = '/home/pi/audiocfg.ini'

# read .ini according to section
def read_config(r_cfg_file, section):
    #print("Read config from " + os.path.abspath(r_cfg_file))
    config = configparser.ConfigParser()
    config.read(r_cfg_file, encoding='utf-8')
    config_dict = dict(config.items(section))
    # print(config.sections())
    '''
    print('\033[1;32m')
    print('{0} = {1}'.format(section, config_dict))
    print('\033[0m')
    '''
    return config_dict

url = read_config(cfg_file,'url')
dir = read_config(cfg_file,'dir')
id  = read_config(cfg_file,'fj_id')
myserial = read_config( cfg_file,'myserial')


def postmydata(value,createTime,sensorId):
    global uploadTime
    try:
        def wirtedata(value,createTime,sensorId):
            headers = ['type', 'value', 'createTime', 'sensorId']
            rows = [
                value,
                createTime,
                sensorId
            ]
            print('write rows to csv',rows)
            dbcsv_dir = dir['dbcsv_dir']
            with open(dbcsv_dir, 'a')as f:
                f_csv = csv.writer(f)
                #f_csv.writerow(headers)
                f_csv.writerow(rows)

        #url = 'http://192.168.199.128:12521/audio-db-data/dbDataupload.do'
        posturl = url['httppost_url']
        #posturl = 'http://192.168.199.128:12521/audio-db-data/dbDataupload.do'
        OriginData = {
                "createTime": createTime,
                "sensorId": sensorId,
                "value": value
            }
        print(OriginData)
        try:
            resp = requests.post(posturl, json = OriginData,timeout=5)
        except:
            # upload failed ,save data to csv
            print('\033[1;35mupload failed ,save to csv \033[0m')
            wirtedata(value, createTime, sensorId)

        if(resp.status_code == 200):
            print('\033[1;36m upload success!! \033[0m')
        if(resp.status_code != 200):
            #upload failed ,save data to csv
            print('\033[1;35mupload failed ,save to csv \033[0m')
            wirtedata(value,createTime,sensorId)
        json = resp.json()
        print(json)
    except BaseException as e:
        print(e)


com = myserial['com']
#com = '/dev/ttyS2'
bsp = myserial['bsp']
#bsp = 4800

x=serial.Serial(com,bsp)#这是我的串口，测试连接成功，没毛病
# i=0
def getdata():#发送函数

    myinput= bytes([0X01,0X03,0X00,0X00,0X00,0X01,0X84,0X0A])
    #这是我要发送的命令，原本命令是：01 03 00 00 00 01 84 0A
    x.write(myinput)

    myout=x.read(7)#读取串口传过来的字节流，这里我根据文档只接收7个字节的数据
    datas =''.join(map(lambda x:('/x' if len(hex(x))>=4 else '/x0')+hex(x)[2:],myout))#将数据转成十六进制的形式
    new_datas = datas.split("/x")#将字符串分割，拼接下标4和5部分的数据
    need = new_datas[4]+new_datas[5];#need是拼接出来的数据，比如：001a
    my_need = int(hex(int(need,16)),16)#将十六进制转化为十进制
    my_need = my_need/10
    print('DB',my_need)

    sensorId = id['id']
    #sensorId = '485dbtest'
    postmydata(my_need,str(int(time.time())),sensorId)

if __name__== '__main__':
    getdata()