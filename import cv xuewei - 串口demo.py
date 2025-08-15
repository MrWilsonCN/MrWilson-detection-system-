import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import serial
import time

# 初始化 MediaPipe 手部模块
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 初始化手部捕捉
cap = cv2.VideoCapture(0)


def get_position(landmarks, idx, kuan, gao):
    return (int(landmarks[idx].x * kuan), int(landmarks[idx].y * gao))                     #获取关键点在图像中的宽度和高度函数，把相对坐标转换为图片像素坐标

def laogongxue(landmarks, width, height):
    wrist = get_position(landmarks, mp_hands.HandLandmark.WRIST, width, height)
    base_fingers = [
        get_position(landmarks, mp_hands.HandLandmark.THUMB_CMC, width, height),           #拇指第一关节高度宽度获取
        get_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_DIP, width, height),    #食指第二关节
        get_position(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_DIP, width, height),   #中指第二关节
        get_position(landmarks, mp_hands.HandLandmark.RING_FINGER_DIP, width, height),     #无名指第二关节
        get_position(landmarks, mp_hands.HandLandmark.PINKY_DIP, width, height)            #小指第二关节
    ]

    avg = (sum(p[0] for p in base_fingers) / len(base_fingers), sum(p[1] for p in base_fingers) / len(base_fingers)) #劳宫穴的位置位置描述算法是[0]代表x坐标求平均值[1]代表y坐标再求平均值
    laogongxue_position = (int((wrist[0] + avg[0]) // 2), int((wrist[1] + avg[1]) // 2))
    return laogongxue_position
    

def draw_text(image, text, position, font_path, font_size, color):      #中文字体转换代码函数定义
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   #关键步骤！！！一定要把OpenCV的默认BGR色彩空间，转换为Pillow字体适用的RGB空间格式fromarry函数直接把CV数组转为Poillow
    draw = ImageDraw.Draw(pil_img)                                      

    try:
        font = ImageFont.truetype(font_path, font_size)                 #加载指定路径的字体类型，并且设置好字体大小
    except OSError as e:
        print(f"Error loading font")                                    #如果没有下载好pillow字体则会报错在控制台
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)                    #在图像上进行绘制字体
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)           #最后把pil_img转换为OpenCV格式

if cap.isOpened():  # 判断摄像头是否打开
    ret, frame = cap.read()  # 读取摄像头每一帧的图片，返回一个ret(布尔值),一个图像三维矩阵
    frame = cv2.flip(frame, 1)  # 翻转摄像头图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间格式
    results = hands.process(rgb_frame)  # 将转换好的RGB格式套用到mediapipe上的hands
    height, width, _ = frame.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # 获取劳宫穴的位置坐标
            laogongxu = laogongxue(landmarks, width, height)
            frame = cv2.circle(frame, laogongxu, 10, (255, 0, 0), -1)
            frame = draw_text(frame, '劳宫', (laogongxu[0] + 5, laogongxu[1] - 5),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))
            print(f"劳宫穴 位置: X{laogongxu[0]}Y{laogongxu[1]}")  # 实时打印劳宫穴的像素坐标 
            ser = serial.Serial('COM6',9600,timeout=1)
        while(1):
            if ser.is_open :
                print("串口连接准备就绪！！！")
                data_to_send = f"X{laogongxu[0]}Y{laogongxu[1]}"
                ser.write(data_to_send.encode())
                print(f"{data_to_send}")
                deta2 = ser.readline()
                print(f"{deta2}")
            if deta2 != '0':
                ser.close
                break
            


    cv2.imshow('Hand Tracking', frame)  # 显示图像
cap.release()
cv2.destroyAllWindows()


