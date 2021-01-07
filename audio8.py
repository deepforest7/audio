'''
    功能实现：
        V1.0
            1.声音采集
            2.wav文件上传
        V2.0
            1.开俩个线程实现采集和上传
        V3.0
            1.开线程接入算法
            2.算法结果txt上传
        v5.0
            1.设置下发
        v7.0
            1.挑选最大值上发,如果上发失败，就保存到wav文件中的csv文件
            2.原始数据都保存到wav文件，包括计算后的fengzhi quanzhi dBA AMout等，不上发
            3.nmk采集策略成功。
        v8.0
            1。四声道音频采集。拆分成俩声道。

'''

"""
    流程图
    audio_record.set() 开始录音 -> 录音结束 -> audio_record.set()
                                          -> thread_sf.set()       -> thread_res_post.set()
                                          -> thread_fengzhi.set()
"""

import pyaudio
import wave
import os
import sys
import csv
import configparser
import time
import threading
import requests
from ctypes import *
from tqdm import tqdm
# from CalAMNout2 import CalAMNout2
# github
import soundfile as sf
from numpy import *
import math
import numpy as np
import librosa
import librosa.display

thread_audio_record = None
thread_wav_post = None
thread_sf = None
thread_res_post = None
thread_fengzhi = None

event_wavupload = threading.Event()
audio_record = threading.Event()
event_res_upload = threading.Event()
event_sf = threading.Event()
event_fengzhi = threading.Event()

is_exit = False
cfg_file = './audiocfg.ini'

WAVE_FILENAME = ''
AMNout_filename = ''
dBA_filename = ''
fengzhi_filename = ''
quanzhi_filename = ''




# read .ini according to section
def read_config(r_cfg_file, section):
    print("Read config from " + os.path.abspath(r_cfg_file))
    config = configparser.ConfigParser()
    config.read(r_cfg_file, encoding='utf-8')
    config_dict = dict(config.items(section))
    # print(config.sections())
    print('\033[1;32m')
    print('{0} = {1}'.format(section, config_dict))
    print('\033[0m')
    return config_dict


audiourl = read_config(cfg_file,'http_url')
audiodata = read_config(cfg_file,'data_dir')
fj_id = read_config(cfg_file, 'fj_id')
save_dir = read_config(cfg_file, 'data_dir')
runningtime = read_config(cfg_file, 'runningtime')
RECORD_SECONDS = int(runningtime['record_seconds'])
sleep_time = int(runningtime['sleep_time'])
#单位小时
uploadTime = int(runningtime['uploadtime']) * 10



def postmydata(datatype,datavalue,audioChannel):
    global uploadTime
    try:
        def wirtedata(datatype,datavalue,audioChannel):
            headers = ['type', 'value', 'createTime', 'sensorId']
            rows = [
                datatype,
                datavalue,
                time_now,
                fj_id['id'],
                audioChannel
            ]
            print('rows',rows)
            with open('/home/pi/wav/data.csv', 'a')as f:
                f_csv = csv.writer(f)
                #f_csv.writerow(headers)
                f_csv.writerow(rows)


        json_url = audiourl['res_url']
        #url = 'http://192.168.199.124:12521/laser-data/laserDataupload.do'
        #url1 = url.encode('url-8')
        OriginData = {
                "createTime": time_now,
                "sensorId": fj_id['id'],
                "value": datavalue,
                "type" : datatype,
                "audioChannel" :audioChannel
            }
        print(OriginData)
        resp = requests.post(json_url, json = OriginData,timeout=5)

        if(resp.status_code == 200):
            json = resp.json()
            print(json)
            RECORD_SECONDS = json['data']['collectTime']
            sleep_time = json['data']['collectRange']
            uploadTime = json['data']['uploadTime'] * 10
            print('\033[1;33m post uploadTime = {0}\033[0m'.format(uploadTime))
            print('RECORD_SECONDS', RECORD_SECONDS)
            print('sleep_time', sleep_time)
        else:
            #upload failed ,save data to csv
            print('\033[1;35mupload failed ,save to csv \033[0m')
            wirtedata(datatype,datavalue,audioChannel)
    except BaseException as e:
        print(e)

def slicchannal(filename):
    file1 = (filename)

    f = wave.open(file1, "rb")

    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print(nchannels, sampwidth, framerate, nframes)  # 2 2 44100 11625348
    # 读取波形数据
    str_data = f.readframes(nframes)
    f.close()

    # 将波形数据转换为数组
    wave_data = np.fromstring(str_data, dtype=np.int32)
    wave_data.shape = -1, 2
    wave_data = wave_data.T

    wave_data_1 = wave_data[0]  # 声道1
    wave_data_2 = wave_data[1]  # 声道2

    w1 = wave_data_1.tostring()
    w2 = wave_data_2.tostring()

    # 实现录音
    def record(re_frames, WAVE_OUTPUT_FILENAME):
        """
        :param re_frames: 是二进制的数据
        :param WAVE_OUTPUT_FILENAME: 输出的位置
        :return:
        """
        p = pyaudio.PyAudio()
        CHANNELS = 2
        FORMAT = pyaudio.paInt32
        RATE = framerate  # 这个要跟原音频文件的比特率相同
        print("开始录音")
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(re_frames)
        wf.close()
        print("关闭录音")

    filename13 = save_dir['dir_wav']+'audio13_' + time_now + '_' + fj_id['id'] + '.wav'
    filename24 = save_dir['dir_wav']+'audio24_' + time_now + '_' + fj_id['id'] + '.wav'


    record(w1, filename13)
    record(w2, filename24)
    return filename13,filename24

# record
def audio_record_thread():
    CHUNK = 512
    FORMAT = pyaudio.paInt16
    CHANNELS = 4
    RATE = 16000
    global is_exit
    global WAVE_FILENAME
    global dBA_filename
    global AMNout_filename
    global fengzhi_filename
    global quanzhi_filename
    global time_now
    global filename13
    global filename24
    #30一次上发原始文件 计数器0
    count = 0
    #k值由服务器下发
    while not is_exit:
        try:
            if audio_record.is_set():
                audio_record.clear()
                dll = CDLL("./open.so")
                dll.main()

                dll_reset = CDLL("./reset.so")
                dll_reset.main()

                # start a program and set some paras
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

                # a cache to recive masssage from mircophone
                print("recording...")
                frames = []
                for i in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
                    # print(i/int(RATE / CHUNK * RECORD_SECONDS))
                    data = stream.read(CHUNK)
                    frames.append(data)
                print("done")

                # stop stream, close stream ,close the program
                stream.stop_stream()
                stream.close()
                p.terminate()

                # gain the para about dev from .ini

                time_now = str(int(time.time()))
                print('\033[1;33m{0} \033[0m'.format(time_now))

                WAVE_OUTPUT_FILENAME = save_dir['dir_wav']+'audio_' + time_now + '_' + fj_id['id'] + '.wav'
                dBA_filename = audiodata['dir_wav']+'dBA_' + time_now + '_' + fj_id['id'] + '.txt'
                AMNout_filename = audiodata['dir_wav']+'AMNout_' + time_now + '_' + fj_id['id'] + '.txt'
                fengzhi_filename =audiodata['dir_wav']+ 'fengzhi_' + time_now + '_' + fj_id['id'] + '.txt'
                quanzhi_filename = audiodata['dir_wav']+'quanzhi_' + time_now + '_' + fj_id['id'] + '.txt'

                print(WAVE_OUTPUT_FILENAME)

                # save what record from dev
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                WAVE_FILENAME = WAVE_OUTPUT_FILENAME

                print('声道拆分')
                filename13, filename24 = slicchannal(WAVE_FILENAME)


                print('posting')

                # wake post thread
                count = count +1
                print('\033[1;33m wav post count = {0}/{1}\033[0m'.format(count,uploadTime))

                if count == uploadTime:
                    event_wavupload.set()
                    count =0
                # strat dba and amnout
                event_sf.set()
                # start fengzhi
                event_fengzhi.set()

                dll_close = CDLL("./close.so")
                dll_close.main()

            else:
                audio_record.wait()
        except ValueError as e:
            print("record error")
            event_wavupload.clear()


# when upload setted,start http upload
def wav_post_thread():
    global sleep_time
    global RECORD_SECONDS
    while not is_exit:
        try:
            if event_wavupload.is_set():
                event_wavupload.clear()

                url = audiourl['wav_url']
                #url = 'http://hzrnb.renewp.com:10701/audio-file/fileUpload'

                #url = 'http://192.168.199.149:10701/audio-file/fileUpload'
                files = {'fileName': open(WAVE_FILENAME, 'rb')}
                response = requests.post(url, files=files)
                if response.status_code == 200:
                    json = response.json()
                    print(json)
                    RECORD_SECONDS = json['data']['collectTime']
                    sleep_time = json['data']['collectRange']
                    k = json['data']['uploadTime']
                    print('\033[1;33m post k = {0}\033[0m'.format(k))
                    print('RECORD_SECONDS', RECORD_SECONDS)
                    print('sleep_time', sleep_time)
                    os.remove(WAVE_FILENAME)




            else:
                event_wavupload.wait()
        except BaseException as e:
            print(e)
            event_wavupload.clear()


# CalAMNout2
def CalAMNout2(filename):
    # fileName = 'wstest.wav'
    # fileName = 'test1.wav'
    # fileName = '1596158520_sdrc54350.wav'

    fileName = filename
    print(fileName)
    tPeriod = 5
    startTime = 1
    endTime = 25
    rawData, sampleRate = sf.read(fileName)
    print("采样率：%d" % sampleRate)

    f = wave.open(fileName, 'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("通道：%d" % nchannels)
    print("sampwidth：%d" % sampwidth)
    print("采样率：%d" % framerate)
    print("nframes：%d" % nframes)

    # print(rawData)
    # print(len(rawData))
    totalSec = math.floor(len(rawData) / sampleRate)
    # print(totalSec)
    secData = []
    for i in range(0, totalSec):
        secData.append([])
    # print(secData)
    for i in range(0, totalSec):
        secData[i] = rawData[i * sampleRate:(i + 1) * sampleRate]
    # print(secData)
    # print(len(secData))
    tNum = math.floor(len(secData) / tPeriod) * tPeriod
    j = 1
    dt = 1 / sampleRate
    t1 = []

    for i in arange(dt, tNum, dt):
        t1.append(i * sampleRate * dt)
    startTime = 0
    # print(t1)
    # print(len(t1))
    ps = []
    AMNout = []

    print('first phrase ok!')

    for i in arange(tPeriod, tNum + 1, tPeriod):
        startTime = i - 5
        p1 = []
        for i2 in arange(startTime + 1, i + 1):
            for i3 in arange(0, sampleRate):
                p1.append(secData[i2 - 1][i3][0])
        for m in arange(0, len(p1)):
            ps.append(p1[m])
        '''
        部分A计权滤波计算
        '''
        print('部分A计权滤波计算')
        nti = round(0.03 * sampleRate)
        ns = round(len(p1) / nti)
        dBA = []
        c1 = 3.5041384e+16
        c2 = power(20.598997, 2)
        c3 = power(107.65265, 2)
        c4 = power(737.86223, 2)
        c5 = power(12194.217, 2)
        b = [-1.874614, -2.055522, -0.135209, -0.772908, -0.747052]
        fs = sampleRate
        p1len = len(p1)
        NumUniquePts = int(ceil((p1len + 1) / 2))
        # print(NumUniquePts)
        P1 = np.fft.fft(p1)
        # print(PS)
        # print(len(PS))
        # P1 = []
        # for m in arange(0, NumUniquePts):
        #     P1.append(PS[m])
        f = []
        A = []
        for m in arange(0, NumUniquePts):
            f.append(power((m / 5), 2))
            # print(f[110250])
            # print(len(f))
            num = c1 * power(f[m], 4)
            den = (power((c2 + f[m]), 2) * (c3 + f[m]) * (c4 + f[m]) * power((c5 + f[m]), 2))
            A.append(num / den)
        XA = []
        for m in arange(0, NumUniquePts):
            XA.append(P1[m] * A[m])
        # print(len(XA))
        for m in arange(0, NumUniquePts - 2):
            XA.append(complex(XA[NumUniquePts - m - 2].real, -XA[NumUniquePts - m - 2].imag))
        # print(len(XA))
        SignalA = []
        ifftXA = np.fft.ifft(XA)
        # print(len(ifftXA))
        for m in arange(0, len(XA)):
            SignalA.append(ifftXA[m].real)
        # print(len(SignalA))
        # print(SignalA)
        for m in arange(1, ns - 1):
            tempa = 0
            for m1 in arange(1, nti + 1):
                tempa = tempa + power(SignalA[m * nti + m1 - 1], 2)
            dBA.append(10 * log10(tempa / nti / power(2e-5, 2)) - 24)
        '''
        AMNout计算 
        '''

        print('AMNout计算 ')

        plusdBA = 0
        for m in arange(0, len(dBA)):
            plusdBA = plusdBA + dBA[m]
        dBArange = plusdBA / len(dBA)
        dBAm = []
        for m in arange(0, len(dBA)):
            dBAm.append(dBA[m] - dBArange)
        dt1 = 1 / fs * nti
        M = len(dBAm)
        F = np.fft.fft(dBAm)
        df = 1 / (M * dt1)
        ouT1 = []
        ouT2 = []
        ouT3 = []
        Farange = []
        for m in arange(0, M):
            ouT1.append(df * m)
        for m in arange(0, 83):
            Farange.append(power(F[m].real, 2) + power(F[m].imag, 2))
            ouT2.append(Farange[m] / (M * M * df))
        N = len(ouT2)
        for m in arange(1, 83):
            ouT2[m] = 2 * ouT2[m]
        ouT2[0] = 1.24312086375157e-27
        for m in arange(0, 83):
            ouT2[m] = math.sqrt(2 * ouT2[0] * ouT2[m])
        absouT2 = []
        for m in arange(0, 83):
            absouT2.append(abs(ouT2[m]))
        peak = round(max(absouT2) * power(10, 17), 10) * 10000
        AMNout.append(peak)
        AMNout[j - 1] = AMNout[j - 1] + b[j - 1]
        j = j + 1
        # print(ouT1[164])
        # print(dBA)
        # print(len(dBA))
        # print(f[110250])
        # print(F[110250])
    '''
    整段dBA计算
    '''

    print('整段dBA计算')

    pslen = len(ps)
    ns = round(pslen / nti)
    NumUniquePts = int(ceil((pslen + 1) / 2))
    # print(NumUniquePts)
    PS = np.fft.fft(ps)
    # print(PS)
    # print(len(PS))
    # P1 = []
    # for m in arange(0, NumUniquePts):
    #     P1.append(PS[m])
    f = []
    A = []
    dBA = []
    for m in arange(0, NumUniquePts):
        f.append(power((m / sampleRate * pslen), 2))
        # print(f[110250])
        # print(len(f))
        num = c1 * power(f[m], 4)
        den = (power((c2 + f[m]), 2) * (c3 + f[m]) * (c4 + f[m]) * power((c5 + f[m]), 2))
        A.append(num / den)
    XA = []
    for m in arange(0, NumUniquePts):
        XA.append(PS[m] * A[m])
    for m in arange(0, NumUniquePts - 2):
        XA.append(complex(XA[NumUniquePts - m - 2].real, -XA[NumUniquePts - m - 2].imag))
    # print(len(XA))
    SignalA = []
    ifftXA = np.fft.ifft(XA)
    # print(len(ifftXA))
    for m in arange(0, len(XA)):
        SignalA.append(ifftXA[m].real)
    # print(len(SignalA))
    # print(SignalA)
    for m in arange(1, ns - 1):
        tempa = 0
        for m1 in arange(1, nti + 1):
            tempa = tempa + power(SignalA[m * nti + m1 - 1], 2)
        dBA.append(10 * log10(tempa / nti / power(2e-5, 2)) - 24)
    '''
    绘图
    '''
    #print('绘图')


    return dBA,AMNout

"""
    header = ['dBA_value']
    # dBA_filename = 'dBA_' + time_now + '_' + fj_id['id'] + '.txt'
    print('dBA_filename', dBA_filename)
    with open(dBA_filename, 'w', newline="") as f:  # 保存文件
        for temp in dBA:

            # print(temp)
            dict_dBA = {'dBA_value': temp}
            # print(dict_dBA)

            f_csv = csv.DictWriter(f, header)
            f_csv.writerow(dict_dBA)

    header = ['AMNout_value']
    # AMNout_filename = 'AMNout_' + time_now + '_' + fj_id['id'] + '.txt'
    print('AMNout_filename', AMNout_filename)
    with open(AMNout_filename, 'w', newline="") as f:  # 保存文件
        for temp in AMNout:
            # print(temp)
            dict_AMNout = {'AMNout_value': temp}
            # print(dict_AMNout)

            f_csv = csv.DictWriter(f, header)
            f_csv.writerow(dict_AMNout)"""


# when upload setted,start http upload
def sf_thread():
    while not is_exit:
        try:
            if event_sf.is_set():
                event_sf.clear()

                print('Cal start')
                dBA,AMNout = CalAMNout2(filename13)

                postmydata('dBA', [(max(dBA))],"13")
                postmydata('AMNout', [max(AMNout)],"13")


                dBA,AMNout = CalAMNout2(filename24)
                postmydata('dBA', [(max(dBA))], "24")
                postmydata('AMNout', [max(AMNout)], "24")
                # print(dBA)
                # print("maxdbA \r\n", max(dBA))

                # print(AMNout)
                # print("maxAMNout \r\n", max(AMNout))


                print('Cal end')

                event_res_upload.set()

            else:
                event_sf.wait()
        except BaseException as e:
            print(e)
            event_sf.clear()


# fengzhi
def fengzhi(path, n):  # 音频路径,散点图数量
    global fengzhi_filename
    global quanzhi_filename
    # 读取文件,获得一些音频文件的信息
    wf = wave.open(path, 'rb')
    params = wf.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    # print('音频采样率为:'+str(framerate)+'hz')
    # print('位数为:'+str(sampwidth*8)+'位')
    # print('声道数为:'+str(nchannels))
    str_data = wf.readframes(nframes)
    wf.close()
    wavedata = np.frombuffer(str_data, dtype=np.int16)
    # 划分声道,如果为多声道,转置之后数组就有多个维度
    wavedata.shape = -1, nchannels
    wavedata = wavedata.T
    # 归一化
    wavedata2 = wavedata * 1.0 / np.max(np.abs(wavedata))
    time = np.arange(0, nframes) * (1.0 / framerate)
    y, sr = librosa.load(path, sr=None, mono=True)
    # plt.plot(time, wavedata2[0])

    # plt.xlabel("Time(s)")
    # plt.ylabel("Amplitude")
    # plt.show()
    # 计算频谱和功率谱,可以看下librosa文档,学一下
    c1 = np.abs(np.fft.fft(wavedata2[0]))
    N = len(c1)
    halfN = int(N / 2 + 1)
    freq = np.linspace(0, framerate / 2, halfN)
    db = librosa.amplitude_to_db(c1)
    power = librosa.db_to_power(db)
    db = db[:halfN]
    power = power[:halfN]
    # tem=freq[np.argmax(power)]
    # print('主频为'+str(tem)+'hz')
    # plt.subplot(211)
    # plt.plot(freq, db)
    # plt.xlabel("Frequency(hz)")
    # plt.ylabel("SPL(db)")
    # plt.subplot(212)
    # plt.xlabel("Frequency(hz)")
    # plt.plot(freq, power)
    # plt.show()
    ospl = np.power(10, np.divide(db, 10)).sum()
    ospl = 10 * np.log10(ospl)
    print('总声压为' + str(ospl) + 'db')  # 总声压

    # 剩下就是搞了几个循环算各段的峰值频率,对到对应的数组里
    xlist = []
    ylist = []
    pnly = []
    l = 0
    r = 0
    length = len(db)
    for i in range(1, n + 1, 1):
        if i == n:
            t = power[l:]
            tf = freq[l:]
            b = t[np.argmax(t)]
            pnly.append(tf[np.argmax(t)])
            xlist.append(freq[length - 1])
        else:
            r = int(i * np.floor(length / n))
            xlist.append(freq[r])
            t = power[l:r]
            tf = freq[l:r]
            b = t[np.argmax(t)]
            pnly.append(tf[np.argmax(t)])
        ylist.append(b)
        l = r
    # plt.scatter(xlist, ylist)
    # plt.xlabel("Frequency(hz)")
    # plt.ylabel("SPL(db)")
    # plt.show()
    print('功率谱各段频率的峰值:')  # 功率谱频谱散点图
    print(np.array(pnly))
    print('功率谱各频率的权值:')
    print(np.array(ylist))
    nxlist = []
    nylist = []
    npnly = []
    nl = 0
    nr = 0
    nlength = len(db)
    for i in range(1, n + 1, 1):
        if i == n:
            t = db[nl:]
            tf = freq[nl:]
            b = t[np.argmax(t)]
            npnly.append(tf[np.argmax(t)])
            nxlist.append(freq[length - 1])
        else:
            nr = int(i * np.floor(nlength / n))
            nxlist.append(freq[r])
            t = db[nl:nr]
            tf = freq[nl:nr]
            b = t[np.argmax(t)]
            npnly.append(tf[np.argmax(t)])
        nylist.append(b)
        nl = nr
    # plt.scatter(nxlist, nylist)
    # plt.xlabel("Frequency(hz)")
    # plt.ylabel("SPL(db)")
    # plt.show()
    print('db值各段频率的峰值:')  # db值频谱散点图
    print(np.array(npnly))
    print('db值各频率的权值:')
    print(np.array(ylist))

    # np.seterr(divide='ignore')
    # spectrum, freqs, ts, fig = plt.specgram(y, Fs=framerate, sides='onesided', scale='dB')  # 绘制频谱图
    # plt.colorbar()
    # plt.ylabel('Frequency')
    # plt.xlabel('Time(s)')
    # plt.title('Spectrogram')
    # plt.show()
    # np.seterr(divide='warn')

    np.savetxt(fengzhi_filename, npnly, fmt="%d", delimiter=" ")
    #print('npnly',npnly)
    #print('prenpnly',npnly[0:4])


    print('fengzhi_filename ',fengzhi_filename)
    print('quanzhi_filename ',quanzhi_filename)

    np.savetxt(quanzhi_filename, ylist, fmt="%d", delimiter=" ")
    #print('ylist', ylist)
    #print('preylist', ylist[0:4])

    return npnly,ylist


# fengzhi thread
def fengzhi_thread():
    global tmp1
    global w1
    while not is_exit:
        try:
            if event_fengzhi.is_set():
                event_fengzhi.clear()

                print('fengzhi start')

                # tmp1, w1 = fengzhi(WAVE_FILENAME,10)
                npnly, ylist = fengzhi(filename13, 10)

                postmydata('fengzhi', npnly[0:4],"13")
                postmydata('quanzhi', ylist[0:4],"13")

                tmp1, w1 = fengzhi(filename24, 10)
                postmydata('fengzhi', npnly[0:4], "24")
                postmydata('quanzhi', ylist[0:4], "24")

                print('fengzhi end')


                # event_res_upload.set()

            else:
                event_fengzhi.wait()
        except BaseException as e:
            print(e)
            event_fengzhi.clear()


# when upload setted,start http upload
def res_post_thread():
    while not is_exit:
        try:
            if event_res_upload.is_set():
                event_res_upload.clear()
                """
                url = audiourl['audio_url']
                #url = 'http://hzrnb.renewp.com:10701/audio-file/fileUploadTxt'
                #url = 'http://192.168.199.149:10701/audio-file/fileUploadTxt'
                files = {'fileName': open(dBA_filename, 'rb')}
                dBA_response = requests.post(url, files=files)
                if dBA_response.status_code == 200:
                    json = dBA_response.json()
                    print(json)
                    os.remove(dBA_filename)

                files = {'fileName': open(AMNout_filename, 'rb')}
                dBA_response = requests.post(url, files=files)
                if dBA_response.status_code == 200:
                    json = dBA_response.json()
                    print(json)
                    os.remove(AMNout_filename)

                files = {'fileName': open(fengzhi_filename, 'rb')}
                dBA_response = requests.post(url, files=files)
                if dBA_response.status_code == 200:
                    json = dBA_response.json()
                    print(json)
                    os.remove(fengzhi_filename)

                files = {'fileName': open(quanzhi_filename, 'rb')}
                dBA_response = requests.post(url, files=files)
                if dBA_response.status_code == 200:
                    json = dBA_response.json()
                    print(json)
                    os.remove(quanzhi_filename)
"""
                    # os.remove(WAVE_FILENAME)



            else:
                event_res_upload.wait()
        except BaseException as e:
            print(e)
            event_res_upload.clear()


# start fengzhijisuan thread . manified target.
def fengzhi_thread_start():
    global thread_fengzhi
    if thread_fengzhi is None:
        print("thread_fengzhi_start")
        thread_fengzhi = threading.Thread(target=fengzhi_thread)
        thread_fengzhi.setDaemon(True)
        thread_fengzhi.start()


# start record thread . manified target.
def audio_record_thread_start():
    global thread_audio_record
    if thread_audio_record is None:
        print("audio_record_thread_start")
        thread_audio_record = threading.Thread(target=audio_record_thread)
        thread_audio_record.setDaemon(True)
        thread_audio_record.start()


# start wav post thread . manified target .
def wav_post_thread_start():
    global thread_wav_post
    if thread_wav_post is None:
        print("wav_post_thread_start")
        thread_wav_post = threading.Thread(target=wav_post_thread)
        thread_wav_post.setDaemon(True)
        thread_wav_post.start()


# start res post thread . manified target .
def res_post_thread_start():
    global thread_res_post
    if thread_res_post is None:
        print("res_post_thread_start")
        thread_res_post = threading.Thread(target=res_post_thread)
        thread_res_post.setDaemon(True)
        thread_res_post.start()


# start sf thread . manified target .
def sf_thread_start():
    global thread_sf
    if thread_sf is None:
        print("sf_thread_start")
        thread_sf = threading.Thread(target=sf_thread)
        thread_sf.setDaemon(True)
        thread_sf.start()


def run():
    audio_record_thread_start()
    wav_post_thread_start()
    sf_thread_start()
    fengzhi_thread_start()
    res_post_thread_start()

    while not is_exit:
        # start record
        audio_record.set()

        # sleep for 1 h and wake up to restart again.
        print('sleep for %s' % (sleep_time) + '...')
        time.sleep(sleep_time)

run()
