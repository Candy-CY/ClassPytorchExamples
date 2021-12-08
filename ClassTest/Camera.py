# 实时：视频图像采集(opencv)
import cv2
from pylab import *
import ManyImgs

cap = cv2.VideoCapture(0)
es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
background = None
# 从视频流循环帧
while True:
    ret, frame = cap.read()
    #cv2.imshow("Camera", frame) # 显示原始窗口
    #二值化窗口（全局阈值）：超过阈值的值为最大值，其他值是0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #print("threshold value %s" % ret)  #打印阈值，超过阈值显示为白色，低于该阈值显示为黑色
    #cv2.imshow("threshold", binary) #显示二值化图像
    #加入噪声窗口：
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    mean=0
    var=0.005
    #将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
    image = np.array(image/255, dtype=float)
    #创建一个均值为mean，方差为var呈高斯分布的图像矩阵
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    #将噪声和原始图像进行相加得到加噪后的图像
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #解除归一化，乘以255将加噪后的图像的像素值恢复
    out = np.uint8(out*255)
    #cv2.imshow("gasuss", out)
    noise = noise*255
    #cv2.imshow('gasuss',out)
    #canny边缘检测：
    original_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # canny(): 边缘检测
    img1 = cv2.GaussianBlur(original_img,(3,3),0)
    canny = cv2.Canny(img1, 50, 150)
    #cv2.imshow('Canny', canny)
    # 读取视频流
    grabbed, frame_lwpCV = cap.read()
    # 对帧进行预处理，先转灰度图，再进行高斯滤波。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
    # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。
    #gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(frame_lwpCV, (21,21), 0)
    grabbed, frame_lwpCV1 = cap.read()
    #gray_lwpCV1 = cv2.cvtColor(frame_lwpCV1, cv2.COLOR_BGR2GRAY)
    gray_lwpCV1 = cv2.GaussianBlur(frame_lwpCV1, (21,21), 0)
    '''
    if background is None:
        background = gray_lwpCV
        continue
    '''
    # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
    # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来腐蚀、膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理。
    #diff = cv2.absdiff(background, gray_lwpCV)
    diff = cv2.absdiff(gray_lwpCV, gray_lwpCV1)
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1] # 二值化阈值处理
    diff = cv2.erode(diff,es, iterations=2)
    diff = cv2.dilate(diff, es, iterations=2) # 形态学膨胀
    #cv2.imshow('dis', diff)
    key = cv2.waitKey(1) & 0xFF
    #融合窗口
    #imgs = np.hstack((frame,binary,out,canny))
    '''
    print(frame.shape)
    print(binary.shape)
    print(out.shape)
    print(canny.shape)
    '''
    #cv2.imshow("mutil_pic", imgs)
    #stackedimageh = ManyImgs(0.2, ([frame,binary,out,canny]))
    # 如果只有图片数量为奇数，并希望能够垂直显示，可以创建一个空白图像
    Blankimg = np.zeros((200, 200), np.uint8)  # 大小可以任意函数会将其强制转换
    #stackedimageb = ManyImgs(0.8, ([frame,binary],[out,canny]))
    stackedimageb = ManyImgs.ManyImgs(0.8, ([frame,diff],[out,canny]))
    cv2.imshow("Mix Picture", stackedimageb)
    # 退出：Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 清理窗口
cap.release()
cv2.destroyAllWindows()
