import cv2
import numpy as np
#Shi-Tomas角点检测
def process(image, opt=1):
    # Detecting corners
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 35, 0.05, 10)
    #print(len(corners))
    for pt in corners:
        #print(pt)
        b = np.random.random_integers(0, 256)
        g = np.random.random_integers(0, 256)
        r = np.random.random_integers(0, 256)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv2.circle(image, (x, y), 5, (int(b), int(g), int(r)), 2)
    # output
    return image
#Harris
def harris_det(img, block_size=3, ksize=3, k=0.04, threshold = 0.01, WITH_NMS = False):
    '''
    params:
        img:单通道灰度图片
        block_size:权重滑动窗口
        ksize：Sobel算子窗口大小
        k:响应函数参数k
        threshold:设定阈值
        WITH_NMS:非极大值抑制
    return：
        corner：角点位置图，与源图像一样大小，角点处像素值设置为255
    '''
    h, w = img.shape[:2]
    # 1.高斯权重
    gray = cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=2)
    # 2.计算梯度
    grad = np.zeros((h,w,2),dtype=np.float32)
    grad[:,:,0] = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize=3)
    grad[:,:,1] = cv2.Sobel(gray,cv2.CV_16S,0,1,ksize=3)
    # 3.计算协方差矩阵
    m = np.zeros((h,w,3),dtype=np.float32)
    m[:,:,0] = grad[:,:,0]**2
    m[:,:,1] = grad[:,:,1]**2
    m[:,:,2] = grad[:,:,0]*grad[:,:,1]
    m = [np.array([[m[i,j,0],m[i,j,2]],[m[i,j,2],m[i,j,1]]]) for i in range(h) for j in range(w)]
    # 4.计算局部特征结果矩阵M的特征值和响应函数R(i,j)=det(M)-k(trace(M))^2  0.04<=k<=0.06
    D,T = list(map(np.linalg.det,m)),list(map(np.trace,m))
    R = np.array([d-k*t**2 for d,t in zip(D,T)])
    # 5.将计算出响应函数的值R进行非极大值抑制，滤除一些不是角点的点，同时要满足大于设定的阈值
    #获取最大的R值
    R_max = np.max(R)
    #print(R_max)
    #print(np.min(R))
    R = R.reshape(h,w)
    corner = np.zeros_like(R,dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if WITH_NMS:
                #除了进行进行阈值检测 还对3x3邻域内非极大值进行抑制(导致角点很小，会看不清)
                if R[i,j] > R_max*threshold and R[i,j] == np.max(R[max(0,i-1):min(i+2,h-1),max(0,j-1):min(j+2,w-1)]):
                    corner[i,j] = 255
            else:
                #只进行阈值检测
                if R[i,j] > R_max*threshold :
                    corner[i,j] = 255
    return corner

