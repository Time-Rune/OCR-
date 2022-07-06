"""
将一行文本剪切为多个单词的形式，返回整行的识别效果
"""
import cv2
import numpy as np
from Predict import Predict
from Sauvola import sauvola


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


def show(img_show):
    cv2.imshow("test", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut(img):
    # 经典转灰度图+二值化
    img2 = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # binary = sauvola(binary)

    # show(binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.fillPoly(binary, [cnt], (255, 255, 255))

    # 首先确定膨胀和腐蚀的核函

    binary = cv2.GaussianBlur(binary, (3, 3), 0)
    # show(binary)

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4))

    for i in range(2):
        binary = cv2.dilate(binary, element1, iterations=2)
        binary = cv2.erode(binary, element2, iterations=2)
    binary = cv2.dilate(binary, element1, iterations=1)

    # show(binary)

    # 再取一次轮廓，切割出图形
    parts = []
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        # 删掉太小的矩形
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        # 将方框画在原图上
        box = cv2.boxPoints(rect)
        # print("当前矩形为： ", box)
        box = np.int0(box)

        board = Get_counter(box, len(img), len(img[0]))
        parts.append(board)
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        cv2.drawContours(img2, [cnt], -1, (0, 255, 0), 2)
    parts.sort(key=lambda board: board[2])
    Answer = ''
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for board in parts:
        Answer += Predict(img[:, board[2]:board[3]])
        Answer += ' '
    # print(Answer)
    show(img2)
    return Answer
    # show(img)


if __name__ == '__main__':
    IMG = '/Users/rune/PycharmProjects/RNN_book/Test_cv/Part/22.jpg'
    # img = cv2.imread(IMG)
    # cut(img)
    # for i in range(62):
    #     img_path = IMG + ("%d.jpg" % i)
    #     img = cv2.imread(img_path)
    #     cut(img)
    img = cv2.imread(IMG)
    print("Predict:" + cut(img))
    show(img)
