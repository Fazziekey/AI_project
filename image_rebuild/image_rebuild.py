# Developer：Fazzie
# Time: 2021/4/2619:35
# File name: image_rebuild.py
# Development environment: Anaconda Python

from matplotlib import pyplot as plt  # 展示图片
import math
import numpy as np  # 数值处理
from numpy import random
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析
from skimage.measure import compare_ssim as ssim
from scipy import spatial
from PIL import Image


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片

    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)

    # 图像数组数据放缩在 0-1 之间
    # return image.astype(np.double) / info.max
    return image.astype(np.double) / 255


def noise_mask_image(img, noise_ratio=[0.8, 0.4, 0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------

    # 获取矩阵维度
    width = img.shape[0]
    lenth = img.shape[1]

    # 生成各通道滤波矩阵
    red = np.random.choice([0, 1], size=(width, lenth), p=[noise_ratio[0], 1 - noise_ratio[0]])
    blue = np.random.choice([0, 1], size=(width, lenth), p=[noise_ratio[1], 1 - noise_ratio[1]])
    yellow = np.random.choice([0, 1], size=(width, lenth), p=[noise_ratio[2], 1 - noise_ratio[2]])

    # 滤波器叠加
    filters = np.array([red, blue, yellow])
    filters = np.transpose(filters, axes=[1, 2, 0])
    filters.swapaxes(1, 0)

    # 噪声图片生成
    noise_img = img * filters

    # -----------------------------------------------

    return noise_img


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像
    :param img:原始图像
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0

    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)

    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None

    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))

    return round(error, 3)


def calc_ssim(img, img_noise):
    """
    计算图片的结构相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    return ssim(img, img_noise,
                multichannel=True,
                data_range=img_noise.max() - img_noise.min())


def calc_csim(img, img_noise):
    """
    计算图片的 cos 相似度
    :param img: 原始图片， 数据类型为 ndarray, shape 为[长, 宽, 3]
    :param img_noise: 噪声图片或恢复后的图片，
                      数据类型为 ndarray, shape 为[长, 宽, 3]
    :return:
    """
    img = img.reshape(-1)
    img_noise = img_noise.reshape(-1)
    return 1 - spatial.distance.cosine(img, img_noise)


def read_img(path):
    img = Image.open(path)
    img = img.resize((150, 150))
    img = np.asarray(img, dtype="uint8")
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(img.dtype)
    # 图像数组数据放缩在 0-1 之间
    return img.astype(np.double) / info.max


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)
    res_img.astype(np.float64)
    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)


    # -------------实现图像恢复代码答题区域----------------------------

    '''
    r, g, b = cv2.split(noise_mask)
    noise_mask = r * g * b
    noise_mask = noise_mask.astype(np.uint8)
    res_img = cv2.inpaint((noise_img * 255).astype(np.uint8), noise_mask, 3, cv2.INPAINT_TELEA)
    '''
    # 高斯滤波
    #res_img = cv2.GaussianBlur(noise_img, (3, 3), 0)


    # 均值滤波
    #res_img = cv2.blur(noise_img, (5, 5))


    # 中值滤波
    # res_img = cv2.medianBlur(noise_img, 3)


    # 线性回归
    #for ss in range(2,6):
        #restore_part(ss, res_img)

    res_img = mean_filter(res_img)
    res_img = cv2.GaussianBlur(res_img, (3, 3), 0)

    # ---------------------------------------------------------------

    return res_img

def mean_filter(img):
    width = img.shape[0]
    lenth = img.shape[1]
    dim = img.shape[2]
    ss = 1

    for tx in range(ss,width-ss):
        for ty in range(ss,lenth-ss):
            for k in range(dim):
                if img[tx][ty][k] == 0:
                    window = list()
                    for row in range(tx-ss, tx + ss):
                        for col in range(ty-ss, ty + ss):
                            if img[row][col][k] != 0:
                                window.append(img[row][col][k])
                    window = np.array(window)
                    if window.size != 0:
                        img[tx][ty][k] = window.mean()
    return img

def get_data(tx, ty, ss, img, k):
    mid = ss/2
    data = list()
    sample = list()
    count = 0
    for rid in range(tx, tx+ss):
        for cid in range(ty, ty+ss):
            nx = rid - tx
            ny = cid - ty
            nx /= (ss - 1)
            ny /= (ss - 1)
            nx = math.exp(-((nx - mid) ** 2) / 2)
            ny = math.exp(-((ny - mid) ** 2) / 2)
            if img[rid][cid][k] != 0:
                data.append([nx, ny, img[rid][cid][k]])
                count = count + 1
            sample.append([nx, ny])
    data = np.array(data)
    sample = np.array(sample)

    return data, sample


def train(sample, data, tx, ty, ss, img, k):
    # 如果目标区域像素全部损坏，则跳过
    if data.shape == 0:
        return 0
    # 数据加载

    data_x = data[:, 0:2]
    data_y = data[:, 2]

    # 模型加载
    model = LinearRegression()

    # 模型的训练
    model.fit(data_x, data_y)

    # 原图拟合
    result = model.predict(sample)


    num = 0
    for i in range(ss):
        for j in range(ss):
            img[tx+i][ty+j][k] = result[num]
            num = num + 1
    print("row:", tx, "col:", ty, "dim:", k, "has train", "score=", model.score(data_x, data_y))
    return 0


def restore_part(ss, img):
    width = img.shape[0]
    lenth = img.shape[1]
    dim = img.shape[2]

    for row in range(width-ss):
        for col in range(lenth-ss):
            for k in range(dim):
                data, sample = get_data(row, col, ss, img, k)
                train(sample, data, row, col, ss, img, k)

    return img



    # 原始图片
# 加载图片的路径和名称
img_path = 'A.png'

# 读取原始图片
img = read_image(img_path)

# 展示原始图片
plot_image(image=img, image_title="original image")

# 生成受损图片
# 图像数据归一化
nor_img = normalization(img)

# 每个通道数不同的噪声比率
noise_ratio = [0.4, 0.6, 0.8]


# 生成受损图片
noise_img = noise_mask_image(nor_img, noise_ratio)

if noise_img is not None:
    # 展示受损图片
    plot_image(image=noise_img, image_title="the noise_ratio = %s of original image" % noise_ratio)
    # 恢复图片
    res_img = restore_image(noise_img)

    # 计算恢复图片与原始图片的误差
    print("恢复图片与原始图片的评估误差: ", compute_error(res_img, nor_img))
    print("恢复图片与原始图片的 SSIM 相似度: ", calc_ssim(res_img, nor_img))
    print("恢复图片与原始图片的 Cosine 相似度: ", calc_csim(res_img, nor_img))

    # 展示恢复图片
    plot_image(image=res_img.astype(np.float64), image_title="restore image")

    # 保存恢复图片
    save_image('res_' + img_path, res_img)
else:
    # 未生成受损图片
    print("返回值是 None, 请生成受损图片并返回!")



