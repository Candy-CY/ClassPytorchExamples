import cv2
import CDoperation
from PCV.localdescriptors import sift
camera = cv2.VideoCapture(0)
if __name__ == '__main__':
    while True:
        ret, frame = camera.read()
        #Shi-Tomas角点检测
        result = CDoperation.process(frame)
        #cv2.imshow('Shi-Tomas', result)
        #Harris角点检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dst = cv2.cornerHarris(gray, blockSize=3, ksize=5, k=0.05)
        #dst = CDoperation.harris_det(gray)
        image_dst = frame[:, :, :]
        image_dst[dst > 0.01 * dst.max()] = [255,0,0]
        cv2.imshow('Harris',image_dst)
        # 退出：Q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # 清理窗口
    camera.release()
    cv2.destroyAllWindows()

