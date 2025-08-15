import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

def calculate_shao_fu_position(landmarks, width, height):
    index_finger_mcp = get_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_MCP, width, height)                #少府穴的位置是食指指根
    shao_fu_position = (int(index_finger_mcp[0]), int(index_finger_mcp[1]))                            
    return shao_fu_position


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

while cap.isOpened():                                                   #判断摄像头是否打开？
    ret, frame = cap.read()                                             #读取摄像头每一帧的图片返回一个布尔值
    if not ret:
        break                                                           #如果读取失败跳出循环
    frame = cv2.flip(frame, 1)                                          #翻转摄像头图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                  #转换颜色空间格式
    results = hands.process(rgb_frame)                                  #将转换好的RGB格式套用到mediapipe上的han
    height, width, _ = frame.shape       
                                   

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, width, height) #获取中冲穴的位置坐标并使用RGB信号来传导
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)                                   #设定字符显示在穴位的下方位置
            frame = draw_text(frame, '中冲', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),#           #执行代码
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))
            
            laogongxu = laogongxue(landmarks, width, height)                                                    #劳宫穴位置获取（宽度，高度）
            frame = cv2.circle(frame, laogongxu, 10, (255, 0, 0), -1)                                           #运用了劳宫穴函数确定劳宫穴的xy坐标
            frame = draw_text(frame, '劳宫', (laogongxu[0] + 10, laogongxu[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))
            
            shao_fu_position = calculate_shao_fu_position(landmarks, width, height)
            frame = cv2.circle(frame, shao_fu_position, 10, (0, 0, 255), -1)
            frame = draw_text(frame, '少府', (shao_fu_position[0] + 10, shao_fu_position[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, width, height)
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)
            frame = draw_text(frame, '前头', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))  

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, width, height)
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)
            frame = draw_text(frame, '偏头', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))  

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.THUMB_TIP, width, height)
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)
            frame = draw_text(frame, '全头', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))  

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.PINKY_TIP, width, height)
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)
            frame = draw_text(frame, '后头', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))  

            middle_finger_tip = get_position(landmarks, mp_hands.HandLandmark.WRIST, width, height)
            frame = cv2.circle(frame, middle_finger_tip, 10, (0, 255, 0), -1)
            frame = draw_text(frame, '大陵', (middle_finger_tip[0] + 10, middle_finger_tip[1] - 10),
                              'C:\\Windows\\Fonts\\simhei.ttf', 24, (255, 255, 255))
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
