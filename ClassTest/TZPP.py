import cv2
import numpy as np
import copy

class PicDealDraw(object):
    def __init__(self, img2, img1):
        self.img1 = img1
        self.img2 = img2
    def picSiftPicLinker(self): #3.4.2.16一下
        sift_base = cv2.SIFT_create()  # 创建特征检测器,用于检测模板和图像上的特征点
        kp1, des1 = sift_base.detectAndCompute(self.img1, None)  # 获取特征点和特征描述符
        kp2, des2 = sift_base.detectAndCompute(self.img2, None)
        bf = cv2.BFMatcher(crossCheck=True)  # 创建暴力匹配
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)  # python中sorted()：根据x.distance进行排序
        sift_img = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches[0:10], None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        return sift_img

# brief:角点检测
def harrisCorner(gray):
    gray = copy.deepcopy(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    frame[dst > 0.01 * dst.max()] = [0, 0, 255]
    return frame

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    _, frame_di = capture.read()
    while True:
        # prepare data
        _, frame = capture.read()
        if np.all(frame_di == frame) == True:
            print("no difference between two pic ")  ###证明随着时间的不同，采集到的图片是不一样的
        gray1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        harris_fea = harrisCorner(gray1)  # harrisCorner
        cv2.imshow("Harris_Fea:",harris_fea)
        frame_di_obj = PicDealDraw(frame,frame_di)
        two_img_fea = frame_di_obj.picSiftPicLinker()
        cv2.imshow("SIFT Match:", two_img_fea)
        frame_di = frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            capture.release()
            break
    cv2.waitKey(0)
    cv2.destoryAllWindows()
