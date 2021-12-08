'''
import cv2
import numpy as np
camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
# 判断视频是否打开
if (camera.isOpened()):
  print('摄像头成功打开')
else:
  print('摄像头未打开')
# 测试用,查看视频size
size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:'+repr(size))
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
while True:
  # 读取视频流
  grabbed, frame_lwpCV = camera.read()
  # 对帧进行预处理，先转灰度图，再进行高斯滤波。
  # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
  gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
  gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
  # 将第一帧设置为整个输入的背景
  if background is None:
    background = gray_lwpCV
    continue
  # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
  # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
  diff = cv2.absdiff(background, gray_lwpCV)
  diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
  diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
  # 显示矩形框
  contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
  for c in contours:
    if cv2.contourArea(c) < 1500: # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
      continue
    (x, y, w, h) = cv2.boundingRect(c) # 该函数计算矩形的边界框
    cv2.rectangle(frame_lwpCV, (x, y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow('contours', frame_lwpCV)
  #cv2.imshow('dis', diff)
  key = cv2.waitKey(1) & 0xFF
  # 按'q'健退出循环
  if key == ord('q'):
    break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()
'''

import numpy as np
import cv2
import time
import datetime
cap = cv2.VideoCapture(0)#打开一个视频

# ShiTomasi 角点检测参数
feature_params = dict( maxCorners = 100,qualityLevel = 0.3,
                       minDistance = 7,blockSize = 7 )
# lucas kanade光流法参数
lk_params = dict( winSize  = (15,15),maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# 创建随机颜色
color = np.random.randint(0,255,(100,3))
# 获取第一帧，找到角点
ret, old_frame = cap.read()
#找到原始灰度图
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#获取图像中的角点，返回到p0中
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# 创建一个蒙版用来画轨迹
mask = np.zeros_like(old_frame)
while(1):
    ret,frame = cap.read() #读取图像帧
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #灰度化
    # 计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # 选取好的跟踪点
    good_new = p1[st==1]
    good_old = p0[st==1]
    # 画出轨迹
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b= new.ravel()#多维数据转一维,将坐标转换后赋值给a，b
        a=int(a)
        b=int(b)
        c,d = old.ravel()
        c=int(c)
        d=int(d)
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)#画直线
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)#画点
    img = cv2.add(frame,mask) # 将画出的线条进行图像叠加
    cv2.imshow('frame',img)  #显示图像
    k = cv2.waitKey(30) & 0xff #按Esc退出检测
    if k == 27:
        break
    # 更新上一帧的图像和追踪点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
#关闭所有窗口
cap.release()
cv2.destroyAllWindows()
