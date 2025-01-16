import math
import random
import time
import numpy as np


# inverseCDFTheta函数：根据CDF表和随机数反推角度
def inverse_cdf_theta(cdf_table, u):
    """
    生成角度的分布
    :param cdf_table: 角度的 CDF 分布 [(CDF值, 角度值), ...]
    :param u: 随机数 (0到1之间)
    :return: 计算得到的角度
    """
    random_value = u
    left, right = 0, len(cdf_table) - 1

    if right < 0:
        raise ValueError("CDF table cannot be empty")  # 确保 cdf_table 非空

    while right - left > 1:
        mid = (left + right) // 2
        if cdf_table[mid][0] < random_value:
            left = mid
        else:
            right = mid

    # 线性插值计算角度
    t = (random_value - cdf_table[left][0]) / (cdf_table[right][0] - cdf_table[left][0])
    return cdf_table[left][1] + t * (cdf_table[right][1] - cdf_table[left][1])


# GaisserTang函数：根据给定的E0和theta值计算Gaisser-Tang分布
def gaisser_tang(E0, theta, GeV=1.0):
    """
    Gaisser-Tang分布计算函数
    :param E0: 能量 (单位：GeV)
    :param theta: 角度 (单位：弧度)
    :param GeV: 能量单位，默认为1GeV
    :return: 返回概率密度函数值
    """
    A_T = 1.0  # 归一化系数，根据实际模拟需求调整
    r_c = 0.0  # 任意添加的常数项，视需要调整
    gamma = 2.7  # 衰减系数

    E0_hat = E0 / GeV  # 将能量转换为GeV

    # Gaisser-Tang分布相关参数
    p1, p2, p3, p4, p5 = 0.102573, -0.068287, 0.95, 0.0407253, 0.8

    # 计算cos(theta)
    x = np.cos(theta)
    numerator = np.sqrt(x**2 + p1**2 + p2 * x**p3 + p4 * x**p5)
    denominator = np.sqrt(1 + p1**2 + p2 + p4)
    cos_theta_star = numerator / denominator

    # 根据条件调整能量
    if 100 * GeV / cos_theta_star > E0 > 1 / cos_theta_star * GeV:
        Delta = 2.06e-3 * ((950 / cos_theta_star) - 90) / 4.5 * (E0 / GeV * cos_theta_star)
        E0_hat += Delta
        r_c = 1e-4
        A_T = 1.1 * (90 * np.sqrt(np.cos(theta) - 0.001) / 1030)
    elif E0 < 1 / cos_theta_star * GeV:
        E0_hat = (3 * E0 - 7 * (1 / cos_theta_star)) / 10

    term1 = 1.0 / (1.0 + 1.1 * E0_hat * cos_theta_star / 115.0)
    term2 = 0.054 / (1.0 + 1.1 * E0_hat * cos_theta_star / 810.0)

    return A_T * 0.14 * (E0 / GeV) ** -gamma * (term1 + term2 + r_c)


# sampleEnergyFromGaisserTang函数：根据theta采样能量
def sample_energy_from_gaisser_tang(theta, GeV=1.0):
    """
    使用接受-拒绝方法或其他数值抽样技术，根据Gaisser-Tang分布采样能量
    :param theta: 角度 (单位：弧度)
    :param GeV: 能量单位，默认为1GeV
    :return: 采样的能量值
    """
    rand_generator = random.Random(time.time())  # 创建随机数生成器

    while True:
        M = 10
        E = 10 * rand_generator.uniform(100 * 1e6, 1000 * 1e9)  # 生成候选能量（单位：MeV到GeV）

        # 计算概率
        probability = gaisser_tang(E, theta, GeV)
        rand_prob = rand_generator.uniform(0, 1)  # 生成一个 [0, 1] 范围内的随机数

        # 接受-拒绝采样
        if rand_prob < probability / (M * 1.0002 * 10e-6):
            return E


# 示例：根据CDF表和随机数生成角度
cdf_table = [(0.0, 0.0), (0.1, 10.0), (0.2, 20.0), (0.3, 30.0)]  # 示例CDF表格
u = random.random()  # 生成一个[0, 1]范围的随机数
theta = inverse_cdf_theta(cdf_table, u)
print(f"生成的角度：{theta}")

# 示例：采样能量
theta_value = 0.5  # 角度（单位为弧度）
energy = sample_energy_from_gaisser_tang(theta_value)
print(f"采样的能量：{energy}")
