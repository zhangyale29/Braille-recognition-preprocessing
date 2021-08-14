import cv2 as cv
import numpy as np
import os

# 调整图片大小，提高运行效率
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架
    height_new = 550
    width_new = 450
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new

# 直方图归一化，提高灰度图像对比度
def hist_normalization(img_c, a=0, b=255):
    # 取最大灰度与最小灰度
    c = img_c.min()
    d = img_c.max()

    img_out = img_c.copy()

    # 归一化
    img_out = (b - a) / (d - c) * (img_out - c) + a
    img_out[img_out < a] = a
    img_out[img_out > b] = b
    img_out = img_out.astype(np.uint8)

    return img_out

# 提高图像对比度及亮度
def contrast_img(image, c, b):
    rows, cols, channels = image.shape
    blank = np.zeros([rows, cols, channels], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1-c, b)
    return dst

# 采用局部自适应阈值进行图像二值化处理,并进行形态学处理
def image_pretreatment(image):
    # 灰度处理
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('gray1', gray)
    gray = hist_normalization(gray, a=0, b=255)
    cv.imshow('gray2', gray)
    # 高斯滤波，去除噪声
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow('blur', blur)
    # 自适用阈值二值化
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 8)
    cv.imshow('th', th)
    # 像素取反
    th_not = cv.bitwise_not(th)
    cv.imshow('th_not', th_not)
    # 去除椒盐噪声
    img_gray_blur = cv.medianBlur(th_not, 7)
    cv.imshow('img_gray_1', img_gray_blur)
    # 结构化卷积核，考虑盲点，构建一个圆形内核
    kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))
    # 腐蚀
    erode = cv.erode(img_gray_blur, kernel_erode)
    cv.imshow('erode', erode)
    # 膨胀
    dilate = cv.dilate(erode, kernel_dilate)
    cv.imshow('dilate', dilate)
    # 再去椒盐噪声
    img_gray = cv.medianBlur(dilate, 5)
    cv.imshow('img_gray', img_gray)
    return img_gray


# 水平灰度投影
def gray_projection_horizon(img_gray):
    # 读取灰度图行列
    rows, cols = img_gray.shape[:2]
    # 定义灰度柱
    z = np.zeros([rows], np.uint8)
    # 定义变量
    a = 0
    # 定义投影图
    projectionImage = 255 * np.ones([rows, cols, 1], np.uint8)
    # 水平投影积分
    for y in range(0, rows):
        for x in range(0, cols):
            if img_gray[y, x] == 255:
                a = a + 1
            else:
                continue
        z[y] = a
        a = 0
    # 绘制水平投影积分图
    for y in range(0, rows):
        for x in range(0, z[y]):
            projectionImage[y, x] = 0
    return projectionImage


# 竖直灰度投影
def gray_projection_vertical(img_gray):
    # 读取灰度图行列
    rows, cols = img_gray.shape[:2]
    # 定义灰度柱
    v = np.zeros([cols], np.uint8)
    # 定义储值变量
    a = 0
    # 定义投影图
    projectionImage = 255 * np.ones([rows, cols, 1], np.uint8)
    # 竖直投影积分
    for x in range(0, cols):
        for y in range(0, rows):
            if img_gray[y, x] == 255:
                a = a + 1
            else:
                continue
        v[x] = a
        a = 0
    # 绘制竖直投影积分图
    for x in range(0, cols):
        for y in range(0, v[x]):
            projectionImage[y, x] = 0
    return projectionImage


# 图像自动校正，基于灰度投影积分图
def image_rectification(img_gray, image):
    # 读取灰度图行列
    rows, cols = img_gray.shape[:2]
    # 定义底部灰度比值存储列表
    c = [0] * 40
    # 计算截取底部图像总像素
    p0 = 2 * cols
    q0 = 2 * rows
    # 定义索引变量
    b = 0
    # 确定偏转角度
    for alpha in range(-30, 30, 1):
        # 旋转灰度图
        rotation_factor = cv.getRotationMatrix2D((cols / 2, rows / 2), alpha, 0.8)
        rotate = cv.warpAffine(img_gray, rotation_factor, (cols, rows), borderValue=(149, 149, 149))

        # 竖直灰度投影
        projectionImage_vertical = gray_projection_vertical(rotate)
        # 截取竖直灰度投影积分图底部
        roi_vertical = projectionImage_vertical[:2, :cols]
        # 计算roi区域黑色像素
        p1 = 0
        for x in range(0, 2):
            for y in range(0, cols):
                if roi_vertical[x, y] == 0:
                    p1 = p1 + 1

        # 水平灰度投影
        projectionImage_horizon = gray_projection_horizon(rotate)
        # 截取水平灰度投影积分图底部
        roi_horizon = projectionImage_horizon[:rows, :2]
        # 计算roi区域黑色像素
        q1 = 0
        for y in range(0, 2):
            for x in range(0, rows):
                if roi_horizon[x, y] == 0:
                    q1 = q1 + 1

        # 计算底部灰度比值和
        c[b] = p1 / p0 + q1 / q0
        b = b + 1
    # 检索最小比值和，确定偏转角度
    m = c.index(min(c)) * 1 - 20
    # 图像旋转校正
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), m, 0.8)
    rotate_img = cv.warpAffine(image, M, (cols, rows))
    rotate_img_gray = cv.warpAffine(img_gray, M, (cols, rows))
    return rotate_img, rotate_img_gray


def image_segmentation(imag_gray, image, file_num):
    # 创建存储文件夹
    path = 'G:tutorial/' + str(file_num) + '/'
    # 判断如果文件不存在,则创建
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

    # 校正图像灰度投影积分图
    projectionImage_vertical = gray_projection_vertical(imag_gray)
    cv.imshow('projectionImage_vertical', projectionImage_vertical)
    projectionImage_horizon = gray_projection_horizon(imag_gray)
    cv.imshow('projectionImage_horizon', projectionImage_horizon)
    # 状态变量
    flag = 0
    # x投影黑白交替点列表
    array_v = [0] * 100
    # y投影黑白交替点列表
    array_h = [0] * 100
    # 索引变量
    k, j = 0, 0
    # 读取积分图行列大小
    rows, cols = projectionImage_vertical.shape[:2]
    # 统计x投影黑白交替点横坐标
    for x in range(0, cols):
        if projectionImage_vertical[3, x] == 0 and flag == 0:
            flag = 1
            array_v[k] = x
            k = k + 1
        if projectionImage_vertical[3, x] == 255 and flag == 1:
            flag = 0
            array_v[k] = x
            k = k + 1
    # 计算竖向分割线
    index2 = k // 4
    pos_v = [0 for t in range(0, index2 + 1)]
    weight = [0 for t in range(0, index2)]  # 得到切割横坐标
    pos_v[0] = array_v[0] - (array_v[4] - array_v[3]) // 2
    for i in range(1, index2 + 1):
        pos_v[i] = array_v[4 * i - 1] + (array_v[4 * i] - array_v[4 * i - 1]) // 2
    for e in range(0, index2):
        weight[e] = pos_v[e + 1] - pos_v[e]
    if index2 <= 2:
        weight_avg = sum(weight) / index2
    else:
        weight_avg = (sum(weight) - max(weight) - min(weight)) / (index2 - 2)
    for i in range(1, index2 + 1):
        pos_v[i] = int((pos_v[i - 1] + weight_avg) // 1)

    # 统计y投影黑白交替点横坐标
    for y in range(0, rows):
        if projectionImage_horizon[y, 3] == 0 and flag == 0:
            flag = 1
            array_h[j] = y
            j = j + 1
        if projectionImage_horizon[y, 3] == 255 and flag == 1:
            flag = 0
            array_h[j] = y
            j = j + 1
    # 计算横向分割线
    index1 = j // 6
    pos_h = [0 for t in range(0, index1 + 1)]
    height = [0 for t in range(0, index1)]  # 得到切割纵坐标
    pos_h[0] = array_h[0] - (array_h[6] - array_h[5]) // 2
    for i in range(1, index1 + 1):
        pos_h[i] = array_h[6 * i - 1] + (array_h[6 * i] - array_h[6 * i - 1]) // 2
    for d in range(0, index1):
        height[d] = pos_h[d + 1] - pos_h[d]
    if index1 <= 2:
        height_avg = sum(height) / index1
    else:
        height_avg = (sum(height) - max(height) - min(height)) / (index1 - 2)
    for i in range(1, index1 + 1):
        pos_h[i] = int((pos_h[i - 1] + height_avg) // 1)

    # 切割
    for x in range(0, index2):
        for y in range(0, index1):
            roi = image[pos_h[y]:pos_h[y + 1], pos_v[x]:pos_v[x + 1]]
            # 保存
            cv.imwrite(path + str(y) + "_" + str(x) + ".jpg", roi)


def main(image, i):
    # 截取图像中含盲文部分
    bbox = cv.selectROI(image, False)
    cut = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    img_strong = contrast_img(cut, 1.3, 4)
    cv.imshow('img_strong', img_strong)
    # 图像二值化处理
    img_gray = image_pretreatment(img_strong)
    # 图像自动校正
    rotate_img, rotate_img_gray = image_rectification(img_gray, cut)
    cv.imshow('rotate_img', rotate_img)
    cv.imshow('rotate_img_gray', rotate_img_gray)
    # 图像分割
    image_segmentation(rotate_img_gray, rotate_img, i)


if __name__ == '__main__':
    Circulation_flag = True
    file_n = 0
    while Circulation_flag:
        cv.namedWindow("camera", 1)
        # 开启ip摄像头
        # admin是账号，admin是密码
        video = "http://admin:admin@192.168.43.1:8081"
        capture = cv.VideoCapture(video)

        num = 0
        while True:
            success, img = capture.read()
            img = img[180:520, 180:700]
            cv.imshow("camera", img)

            # 按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
            key = cv.waitKey(10)

            if key == 27:
                # esc键断开连接
                print("esc break...")
                break
            if key == ord(' '):
                cv.imwrite("G:/TEST.jpg", img)
                image_new = img_resize(img)
                break
            if key == ord('q'):
                Circulation_flag = False
                break
        if not Circulation_flag:
            break
        capture.release()
        # image_new = cv.imread("G:/2.jpg")
        main(image_new, file_n)
        file_n = file_n + 1
        cv.waitKey(0)
        # 关闭所有窗口
        cv.destroyAllWindows()


    '''
    # 调用摄像头
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    # 循环拍摄
    while cap.isOpened():
        # 读取每一帧
        ret, frame = cap.read()
        # 翻转成正常视角
        frame = cv.flip(frame, 1)
        cv.imshow("capture", frame)
        # 判断是否正确读取帧
        if ret:
            k = cv.waitKey(10)
            # 按Esc退出程序
            if k == 27:
                break
            # 按回车截取当前帧并对其处理
            elif k == 13:
                image_new = img_resize(frame)
                break
    # 释放摄像
    cap.release()
    main(image_new)
    cv.waitKey(0)
    # 关闭所有窗口
    cv.destroyAllWindows()

src = cv.imread("G:/M2.jpg")
# 调整图像大小
image_new = img_resize(src)
main(image_new)
cv.waitKey(0)
# 关闭所有窗口
cv.destroyAllWindows()
'''
