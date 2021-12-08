import cv2
import os
print("=============================================")
print("=  热键(请在摄像头的窗口使用)：             =")
print("=  z: 更改存储目录                          =")
print("=  x: 拍摄图片                              =")
print("=  q: 退出                                  =")
print("=============================================")
print()
class_name = input("请输入存储目录：")
while os.path.exists(class_name):
    class_name = input("目录已存在！请输入存储目录：")
os.mkdir(class_name)
index = 1
cap = cv2.VideoCapture(0)
width = 640
height = 480
w = 360
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

crop_w_start = (width-w)//2
crop_h_start = (height-w)//2

print(width, height)

while True:
    # get a frame
    ret, frame = cap.read()
    # show a frame
    frame = frame[crop_h_start:crop_h_start+w, crop_w_start:crop_w_start+w]
    frame = cv2.flip(frame,1,dst=None)
    cv2.imshow("capture", frame)
    input = cv2.waitKey(1) & 0xFF
    if input == ord('z'):
        class_name = input("请输入存储目录：")
        while os.path.exists(class_name):
            class_name = input("目录已存在！请输入存储目录：")
        os.mkdir(class_name)
    elif input == ord('x'):
        cv2.imwrite("%s/%d.jpeg" % (class_name, index),
                    cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA))
        print("%s: %d 张图片" % (class_name, index))
        index += 1
    if input == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#SIFT算法
from PIL import Image
from pylab import *
from PCV.localdescriptors import sift
im1f = class_name+'/'+'1.jpeg'
im2f = class_name+'/'+'2.jpeg'
im1 = array(Image.open(im1f).convert('L'))
im2 = array(Image.open(im2f).convert('L'))
sift.process_image(im1f, class_name+'/'+'out_sift_1.txt')
l1, d1 = sift.read_features_from_file(class_name+'/'+'out_sift_1.txt')
figure()
gray()
subplot(121)
sift.plot_features(im1, l1, circle=False)
sift.process_image(im2f, class_name+'/'+'out_sift_2.txt')
l2, d2 = sift.read_features_from_file(class_name+'/'+'out_sift_2.txt')
subplot(122)
sift.plot_features(im2, l2, circle=False)
#matches = sift.match(d1, d2)
matches = sift.match_twosided(d1, d2)
print ('{} matches'.format(len(matches.nonzero()[0])))
figure()
gray()
sift.plot_matches(im1, im2, l1, l2, matches, show_below=True)
show()


