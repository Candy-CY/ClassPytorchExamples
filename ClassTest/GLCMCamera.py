import cv2
import numpy as np
np.set_printoptions(suppress=True)
def glcm(arr, d_x, d_y, gray_level=16):
    '''计算并返回归一化后的灰度共生矩阵'''
    max_gray = arr.max()
    height, width = arr.shape
    # 将uint8类型转换为float64，以免数据失真
    arr = arr.astype(np.float64)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小。
    # 量化后灰度值范围：0 ~ gray_level - 1
    arr = arr * (gray_level - 1) // max_gray
    ret = np.zeros([gray_level, gray_level])
    for j in range(height - abs(d_y)):
        for i in range(width - abs(d_x)):
            rows = arr[j][i].astype(int)
            cols = arr[j + d_y][i + d_x].astype(int)
            ret[rows][cols] += 1
    if d_x >= d_y:
        ret = ret / float(height * (width - 1))  # 归一化, 水平方向或垂直方向
    else:
        ret = ret / float((height - 1) * (width - 1))  # 归一化, 45度或135度方向
    return ret

if __name__=='__main__':
    '''
    归一化时分母值根据角度theta变化，0度或90度时为 height * (width - 1),
    45度或135度时为(height - 1) * (width - 1)
    '''
    fp = r'E:\PictureTest\4.jpeg'
    img = cv2.imread(fp)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像，uint8
    #glcm_0 = glcm(img_gray, 1, 0)  # 水平方向
    glcm_1 = glcm(img_gray, 0, 1)  # 垂直方向
    # glcm_2 = glcm(img_gray, 1, 1)  # 45度方向
    # glcm_3 = glcm(img_gray, -1, 1)  # 135度方向
    print(glcm_1)
