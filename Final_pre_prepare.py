import cv2
import numpy as np
from Sauvola import sauvola
import os
# from Final_Cut_word import cut

# 图片路径
IMG = "img.jpg"


def show(img_show):
    cv2.imshow("test", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Get_counter(Box, Y_max, X_max):
    x_min, y_min = 1e9, 1e9
    x_max, y_max = -1, -1
    for point in Box:
        x_min = min(x_min, point[0])
        x_max = max(x_max, point[0])
        y_min = min(y_min, point[1])
        y_max = max(y_max, point[1])
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, X_max)
    y_max = min(y_max, Y_max)

    return [y_min, y_max, x_min, x_max]


# 读入图片

def recognition(img):
    # img = cv2.imread("Test_images/" + IMG)
    # img = cv2.copyMakeBorder(img, 30, 30, 30, 30, cv2.BORDER_REPLICATE)

    # 转为灰度图并二值化
    # gray = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查看是否二值化过，若已有，则直接读取
    if os.path.exists('binary_img/' + IMG):
        binary = cv2.imread('binary_img/' + IMG, cv2.COLOR_BGR2GRAY)
    else:
        binary = sauvola(gray)
        cv2.imwrite("binary_img/" + IMG, binary)

    show(binary)

    # 进行膨胀-腐蚀-膨胀操作，提取文字的轮廓
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 4))
    element3 = cv2.getStructuringElement(cv2.MORPH_RECT, (32, 2))

    # 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element1, iterations=1)
    # show(dilation)
    # 腐蚀一次，去掉细节
    erosion = cv2.erode(dilation, element2, iterations=1)
    # show(erosion)
    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element3, iterations=2)
    show(dilation2)

    ret, bin_img = cv2.threshold(dilation2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 使用cv2内置的函数求图形轮廓
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 存储所有的矩形，为下一步切割作准备
    boxes = []

    cv2.imwrite("dilation2.jpg", dilation2)
    img2 = cv2.imread("dilation2.jpg")
    for cnt in contours:

        cv2.drawContours(img2, [cnt], -1, (0, 255, 0), 2)

        rect = cv2.minAreaRect(cnt)
        # 删掉太小的矩形
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        # 将方框画在原图上
        box = cv2.boxPoints(rect)
        # print("当前矩形为： ", box)
        box = np.int0(box)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        boxes.append(box)

    parts = []
    for box in boxes:
        board = Get_counter(box, len(img), len(img[0]))
        parts.append(board)

    parts.sort(key=lambda part: part[0])

    # Answer = ''
    # for part in parts:
    #     part_img = binary[part[0]:part[1], part[2]:part[3]]
    #     Answer += cut(part_img) + '\n'
    #
    # print(Answer)

    show(img)
    show(img2)
    # cv2.imwrite("contours.jpg", img)


if __name__ == '__main__':
    img = cv2.imread("/Users/rune/PycharmProjects/RNN_book/Test_cv/Test_images/img.jpg")
    recognition(img)
