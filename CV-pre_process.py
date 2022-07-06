import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

Img_Path = '/Users/rune/PycharmProjects/RNN_book/Test_cv/Test_images/IMG_6511.jpeg'
image = cv2.imread(Img_Path)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

# 由于我们是在高为500的图上做透视变换
# 所以需要通过一个比例因子ratio将找到的框拓展到原图上

gary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 高斯滤波去噪
gary = cv2.GaussianBlur(gary, (5, 5), 0)
edged = cv2.Canny(gary, 75, 200)

print("STEP 1: Edge Detection")
cv2.imshow("Image", image)
cv2.imshow("Gary", gary)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# chain_approx_simple: 只保留方向的终点坐标
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)  # 处理不同版本cv返回轮廓不同的情况

# 按轮廓大小排序，提取出最大的5个轮廓，忽略其他的
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# 找到我们需要的矩形框
screenCnt = []
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:  # 如果是四个点的矩形框
        screenCnt = approx
        break

# 绘制在图像上
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
# warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
# T = threshold_local(warped, 11, offset=10, method="gaussian")
# warped = (warped > T).astype("uint8") * 255

print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(orig, height=650))
cv2.imwrite("/Users/rune/PycharmProjects/Test_cv/Test_images/Warped.jpeg", warped)
cv2.imshow("Scanned", imutils.resize(warped, height=650))
cv2.waitKey(0)
