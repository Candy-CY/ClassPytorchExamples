# 实时：视频图像采集(opencv)
import cv2
import numpy as np
from pylab import *
from skimage.feature import local_binary_pattern
import ManyImgs
import matplotlib.pylab as plt
import GLCMCamera as glcm
cap = cv2.VideoCapture(0)
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius # 领域像素点数
# 从视频流循环帧
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    glcm_1 = glcm.glcm(gray, 0, 1) # 垂直方向
    print("GLMC\n:",glcm_1)
    #二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #canny边缘检测：
    original_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img1 = cv2.GaussianBlur(original_img,(3,3),0)
    canny = cv2.Canny(img1, 50, 150)
    #LBP
    lbp = local_binary_pattern(gray, n_points, radius)
    # print("shape_four:",frame.shape,binary.shape,lbp.shape,canny.shape)
    binary_arr = np.zeros([binary.shape[0],binary.shape[1],3],dtype = int)
    lbp_arr = np.zeros([lbp.shape[0],lbp.shape[1],3],dtype = int)
    canny_arr = np.zeros([canny.shape[0],canny.shape[1],3],dtype = int)
    # print("canny_arr.shape:",canny_arr.shape)
    binary_arr[:,:,0] = binary
    binary_arr[:,:,1] = binary
    binary_arr[:,:,2] = binary
    lbp_arr[:,:,0] = lbp
    lbp_arr[:,:,1] = lbp
    lbp_arr[:,:,2] = lbp
    canny_arr[:,:,0] = canny
    canny_arr[:,:,1] = canny
    canny_arr[:,:,2] = canny
    # print("shape_four_array:",binary_arr.shape,lbp_arr.shape,canny_arr.shape,frame.shape)
    cc = np.vstack((frame,binary_arr))
    dd = np.vstack((lbp_arr,canny_arr))
    final_result_link = np.hstack((cc,dd))
    # print("final_result_link.shape:",final_result_link.shape)
    # print("final-type：",type(final_result_link))
    final_result_link = np.array(final_result_link,dtype = np.uint8)
    final_result_link = cv2.resize(final_result_link,(0,0),fx = 0.5,fy=0.5)
    cv2.imshow("This Result Win:",final_result_link)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
# 清理窗口
cv2.destroyAllWindows()
