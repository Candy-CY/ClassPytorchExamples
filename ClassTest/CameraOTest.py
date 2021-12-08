import cv2
import time
camera = cv2.VideoCapture(0) # 参数0表示第一个摄像头
ticks = time.time()
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
#background = None
grabbed, frame_lwpCV = camera.read()
#gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
gray_lwpCV = cv2.GaussianBlur(frame_lwpCV, (21, 21), 0)
background = gray_lwpCV
while True:
  # 读取视频流
  grabbed, frame_lwpCV = camera.read()
  # 对帧进行预处理，先转灰度图，再进行高斯滤波。
  # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。
  # 对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
  #gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
  gray_lwpCV = cv2.GaussianBlur(frame_lwpCV, (21, 21), 0)
  # 将每次0.5s后的帧设置为整个输入的背景
  if time.time()-ticks > 0.5:
    ticks = time.time()
    grabbed, frame_lwpCV = camera.read()
    #gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(frame_lwpCV, (21, 21), 0)
    background = gray_lwpCV
    continue
  # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
  # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
  diff = cv2.absdiff(background, gray_lwpCV)
  diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
  diff = cv2.erode(diff,es, iterations=2)
  diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
  cv2.imshow('After 0.5S Background Replace', diff)
  key = cv2.waitKey(1) & 0xFF
  # 按'q'健退出循环
  if key == ord('q'):
    break
# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()

