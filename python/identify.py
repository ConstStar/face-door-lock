'''
    仅支持在手机AidLux环境下运行！！！！
'''
import requests
import json
import base64

import time
import datetime
import os
from threading import Thread

import dlib
import numpy as np
import cv2
from cvs import *
import android

import smtplib
from email.header import Header
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

droid = android.Android()

''' 配置 '''

# 通知间隔时长
CONFIG_NOTIFY_TIME = 20

# 在同一个有效时间内 识别次数达到设置次数 就判断为【陌生人】 or 【认识的人】
CONFIG_IDENTIFY_TIME = 15  # 识别有效时间
CONFIG_STRANGER_ANS = 10  # 陌生人识别次数
CONFIG_RECOGNIZE_ANS = 5  # 认识的人识别次数

# 置信度 越小越认识
# 默认认识的人的置信度要低于0.3，越小安全性越高。
# 默认陌生人的置信度要高于0.44
CONFIG_STRANGER_CONFIDENCE = 0.44  # 陌生人置信度
CONFIG_RECOGNIZE_CONFIDENCE = 0.3  # 认识的人置信度

# 开锁秘钥，要与stm32中的秘钥相符
CONFIG_UNLOCK_KEY = "123"

# 邮箱SMTP配置
CONFIG_SMTP_SWITCH = False  # 邮箱通知开关
CONFIG_SMTP_HOST = "smtp.exmail.qq.com"  # SMTP服务器
CONFIG_SMTP_PORT = 465  # SMTP端口
CONFIG_SMTP_SSL = True  # SMTP SSL开关
CONFIG_SMTP_EMAIL = "admin@conststar.cn"  # 邮箱账号
CONFIG_SMTP_PASSWORD = ""  # 邮箱授权码
CONFIG_SMTP_RECEIVER = "1164442003@qq.com"  # 接受通知邮箱账号 tip:建议与发送账号为同一个账号，这样邮件进垃圾箱的概率会更小，想要直接不进垃圾箱也可以直接在邮箱中设置

# QQ机器人配置
CONFIG_QQBOT_SWITCH = False  # QQ通知开关
CONFIG_QQBOT_URL = ""  # cqhttp地址 格式: http://xxx.xxx.xxx.xxx:xxxx
CONFIG_QQBOT_ADMIN = 1164442003  # 管理员QQ号码，也就是当识别到陌生人后发送消息通知这个号码
CONFIG_QQBOT_TOKEN = ""  # cqhttp token

''' ****************** '''


''' 全局变量 '''

stranger_ans = 0  # 记录陌生人识别次数
stranger_time = time.time()  # 记录陌生人识别时间
stranger_imageTime = time.time()  # 用来限制采集陌生人图片的频率
stranger_images = []  # 陌生人图片拍照列表 用于通知冷却后一次性发送多张陌生人图片 格式为[(日期字符串，图片文件)...]

recognize_ans = 0  # 记录认识的人识别次数
recognize_time = time.time()  # 记录认识的人识别时间


if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    print("错误", "shape_predictor_68_face_landmarks.dat 文件不存在")
    droid.notify("shape_predictor_68_face_landmarks.dat 文件不存在")
    exit(-1)

if not os.path.isfile("dlib_face_recognition_resnet_model_v1.dat"):
    print("错误", "dlib_face_recognition_resnet_model_v1.dat 文件不存在")
    droid.notify("dlib_face_recognition_resnet_model_v1.dat 文件不存在")
    exit(-1)

if not os.path.isfile("shape_predictor_68_face_landmarks.dat"):
    print("错误", "label.txt 文件不存在")
    droid.notify("label.txt 文件不存在")
    exit(-1)

if not os.path.isfile("dlib_face_recognition_resnet_model_v1.dat"):
    print("错误", "face_feature_vec.txt 文件不存在")
    droid.notify("face_feature_vec.txt 文件不存在")
    exit(-1)

# 人脸检测数据
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# 人脸特征数据
face_repo = np.loadtxt("face_feature_vec.txt", dtype=float)

# 人脸标签数据
face_labels = open("label.txt", 'r')
label = json.load(face_labels)
face_labels.close()

''' ****************** '''


# 连接串口
def SerialConnect():
    devices = json.loads(droid.startSearchSerialUsb().result)

    while not devices or len(devices) == 0:
        print("等待设备插入...")
        time.sleep(5)
        devices = json.loads(droid.startSearchSerialUsb().result)

    deviceId = devices[0]["deviceId"]
    connectResult = droid.connectSerialUsb(0, deviceId, 9600).result

    while connectResult == "false":
        print("等待给予权限...")
        time.sleep(5)
        connectResult = droid.connectSerialUsb(0, deviceId, 9600).result


# 断开连接串口
def SerialDisconnect():
    droid.disconnectSerialUsb()


# 串口发送字符串
def SerialSendString(data):
    droid.writeSerialData(data.encode().hex())


# 串口接收字符串
def SerialReceiveString():
    datahex = droid.getReceiveData().result
    return binascii.a2b_hex(datahex).decode()


# 发送开锁指令
def sendUnlock():
    global CONFIG_UNLOCK_KEY

    SerialConnect()
    SerialSendString(CONFIG_UNLOCK_KEY)
    SerialDisconnect()


# 发送QQ消息
def sendQQMessage(user_id, message, images):
    global CONFIG_QQBOT_TOKEN

    try:
        for image in images:
            ret, jpg = cv2.imencode(".jpg", image[1])
            encoded_string = base64.b64encode(jpg.tobytes()).decode('utf-8')
            message += f"\n{image[0]}[CQ:image,file=base64://{encoded_string}]"

        params = {
            "user_id": user_id,
            "message": message
            }

        headers = {}

        # 如果token不为空 则放置到请求头中
        if CONFIG_QQBOT_TOKEN:
            headers['Authorization'] = 'Bearer ' + CONFIG_QQBOT_TOKEN

        r = requests.post(CONFIG_QQBOT_URL + "/send_private_msg", params=params, headers=headers, timeout=15)
        if r.json()['retcode'] != 0:
            droid.notify("错误", "发送QQ通知失败", None)
            print("发送QQ通知失败")
    except Exception as ex:
        droid.notify("错误", "发送QQ通知失败", None)
        print("发送QQ通知失败")
        print(repr(ex))


# 发送邮箱信息
def sendEmailMessage(receiver, message, images):
    global CONFIG_SMTP_HOST
    global CONFIG_SMTP_PORT
    global CONFIG_SMTP_SSL
    global CONFIG_SMTP_PASSWORD
    global CONFIG_SMTP_EMAIL

    try:
        emailMessage = MIMEMultipart()
        emailMessage['From'] = Header(f'=?utf-8?B?{base64.b64encode("智能门锁系统".encode()).decode()}=?= <{CONFIG_SMTP_EMAIL}>')
        emailMessage['To'] = Header(f"user <{receiver}>")
        emailMessage['Subject'] = Header("检测到陌生人", 'utf-8')
        emailMessage.attach(MIMEText(message, 'plain', 'utf-8'))  # 通过MIMEApplication构造附件1

        for image in images:
            ret, jpg = cv2.imencode(".jpg", image[1])
            att = MIMEImage(jpg.tobytes())
            att["Content-Type"] = 'application/octet-stream'
            att.add_header('content-disposition', 'attachment', filename=f"{image[0]}.jpg")
            emailMessage.attach(att)

        if CONFIG_SMTP_SSL:
            smtp = smtplib.SMTP_SSL(CONFIG_SMTP_HOST, CONFIG_SMTP_PORT)
        else:
            smtp = smtplib.SMTP(CONFIG_SMTP_HOST, CONFIG_SMTP_PORT)
        smtp.login(CONFIG_SMTP_EMAIL, CONFIG_SMTP_PASSWORD)
        smtp.sendmail(CONFIG_SMTP_EMAIL, receiver, emailMessage.as_string())
        smtp.quit()
    except Exception as ex:
        droid.notify("错误", "发送邮箱通知失败", None)
        print("发送邮箱通知失败")
        print(repr(ex))


# 发送通知给管理员
def sendNotifyAdmin(message, images):
    global CONFIG_QQBOT_SWITCH
    global CONFIG_SMTP_SWITCH
    global CONFIG_QQBOT_ADMIN
    global CONFIG_SMTP_RECEIVER

    if CONFIG_QQBOT_SWITCH:
        sendQQMessage(CONFIG_QQBOT_ADMIN, message, images)

    if CONFIG_SMTP_SWITCH:
        sendEmailMessage(CONFIG_SMTP_RECEIVER, message, images)


# 通知线程 循环判断陌生人图片列表是否有内容
def notifyThread():
    global stranger_images

    global CONFIG_NOTIFY_TIME

    print("通知线程开始运行")
    while True:
        if len(stranger_images) != 0:
            # 发送通知
            sendNotifyAdmin("检测到陌生人", stranger_images)
            stranger_images = []
        time.sleep(CONFIG_NOTIFY_TIME)


# 识别后条件处理操作
def conditionOperation(idnum, confidence, image):
    # global names
    global stranger_time
    global stranger_ans
    global stranger_imageTime
    global recognize_time
    global recognize_ans
    global droid

    global CONFIG_RECOGNIZE_CONFIDENCE
    global CONFIG_STRANGER_CONFIDENCE
    global CONFIG_IDENTIFY_TIME
    global CONFIG_RECOGNIZE_ANS

    if confidence <= CONFIG_RECOGNIZE_CONFIDENCE:
        now_time = time.time()

        print(f"认识的人{label[idnum]}")

        # 不在上一个的时间范围内（超过范围），则重新累识别次数
        if recognize_time + CONFIG_IDENTIFY_TIME < now_time:
            recognize_ans = 1
            recognize_time = time.time()
            return

        recognize_ans += 1
        # 如果识别到认识的人次数不够
        if recognize_ans < CONFIG_RECOGNIZE_ANS:
            return

        name = label[idnum]
        print(f"欢迎回来{name}")
        droid.makeToast(f"欢迎回来{name}")

        # 开锁
        sendUnlock()


    elif confidence >= CONFIG_STRANGER_CONFIDENCE:
        print(f"陌生人{stranger_ans}")
        now_time = time.time()

        # 不在上一个的时间范围内（超过范围），则重新累计识别次数
        if stranger_time + CONFIG_IDENTIFY_TIME < now_time:
            stranger_ans = 1
            stranger_time = time.time()
            return

        stranger_ans += 1
        # 如果识别到陌生人的次数不够
        if stranger_ans < CONFIG_STRANGER_ANS:
            return

        # 限制采集陌生人图片的频率
        if stranger_imageTime + CONFIG_NOTIFY_TIME / 8 < now_time:
            dt_object = datetime.datetime.fromtimestamp(now_time)
            formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')
            stranger_images.append((formatted_time, image))
            stranger_imageTime = now_time

        stranger_ans = 0
        stranger_time = time.time()

        droid.makeToast(f"我不太认识你")
        print(f"我不太认识你")
    else:
        print("不知道认不认识", confidence)


def detect_face(img):
    dets = detector(img, 1)  # 使用检测算子检测人脸，返回的是所有的检测到的人脸区域
    bbs = []

    for d in dets:
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(d.left(), 0)
        bb[1] = np.maximum(d.top(), 0)
        bb[2] = np.minimum(d.right(), img.shape[1])
        bb[3] = np.minimum(d.bottom(), img.shape[0])

        # 筛选符合大小的脸 识别到的脸必须为原图的5分之一大小
        if bb[2] - bb[0] < (img.shape[0] // 5):
            continue
        if bb[3] - bb[1] < (img.shape[1] // 5):
            continue
        bbs.append(bb)

    return bbs


# 识别人脸，如果认识返回下标 如果不认识则返回-1
def find_most_likely_face(face_descriptor):
    face_distance = face_descriptor - face_repo
    euclidean_distance = 0
    if len(label) == 1:
        euclidean_distance = np.linalg.norm(face_distance)
    else:
        euclidean_distance = np.linalg.norm(face_distance, axis=1, keepdims=True)
    min_distance = euclidean_distance.min()
    # if min_distance > CONFIG_FACE_THRESHOLD:
    #     return -1

    index = np.argmin(euclidean_distance)
    return index, min_distance


def recognition(img):
    bbs = detect_face(img)
    for bb in bbs:
        rec = dlib.rectangle(bb[0], bb[1], bb[2], bb[3])
        shape = sp(img, rec)
        face_descriptor = facerec.compute_face_descriptor(img, shape)

        idnum, confidence = find_most_likely_face(face_descriptor)
        conditionOperation(idnum, confidence, img)


# 运行
def run():
    global recognizer
    global face_cascade

    # 运行通知线程
    Thread(target=notifyThread).start()

    # 打开相机
    camid = 1
    cap = cvs.VideoCapture(camid)

    identify_time = 0
    while True:
        frame = cvs.read()
        if frame is None:
            continue
        if camid == 1:
            # frame=cv2.resize(frame,(640,480))
            frame = cv2.flip(frame, 1)

        now_time = int(time.time())
        # 限制识别频率 两秒秒内只能识别一次
        if identify_time != now_time:
            identify_time = now_time
            Thread(target=recognition, args=(frame,)).start()

        cvs.imshow(frame)
        # time.sleep(0.010)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
