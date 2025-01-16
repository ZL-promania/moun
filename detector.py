from typing import List
import numpy as np
import scipy
from scipy.interpolate import interp1d


class Board:
    """Board class representing a detector with efficiency calculations."""

    size_x = 1000  # 探测器的x轴尺寸
    size_y = 100  # 探测器的y轴尺寸

    def __init__(self, x, y, threshold):
        """
        初始化探测器参数
        :param x: 探测器 x 轴位置
        :param y: 探测器 y 轴位置
        :param phi: 板子参考系与 Layer 参考系之间的夹角 (角度制)
        :param threshold: 探测效率阈值
        """
        self.x = x
        self.y = y
        self.threshold = threshold

    def efficiency(self, pos_x):
        """
        计算探测效率。
        :param pos_x: 探测点 x 坐标
        :return: 探测效率值
        """
        raw_x = np.array([100, 300, 500, 700, 900])
        raw_eff = np.array([0.85, 0.9, 0.95, 0.9, 0.85])
        f = interp1d(raw_x, raw_eff, kind="linear", fill_value="extrapolate")
        efficiency_value = f(pos_x)
        return max(efficiency_value, self.threshold)

    def get_efficiency(self, x, y):
        """
        获取探测器的探测效率。
        :param x: 探测点 x 坐标
        :param y: 探测点 y 坐标
        :return: 探测效率值
        """
        # 计算旋转后的坐标
        # rotated_x = (x - self.x) * np.cos(self.phi) - (y - self.y) * np.sin(self.phi)
        return self.efficiency(x)


class Layer:
    space = (1100, 105)

    def __init__(self, z_pos, phi):
        """
        初始化探测层
        :param z_pos: 层的位置（z轴）
        """
        self.board_list: List[Board] = []
        self.z_pos = z_pos
        self.phi = phi

    def arrange_boards(self, rows, cols):
        """
        根据行列数和间距排列板的位置。
        :param rows: 行数
        :param cols: 列数
        :param spacing: 板之间的间距
        """
        for i in range(rows):
            for j in range(cols):
                x_pos = j * self.space[0]
                y_pos = i * self.space[1]
                phi = self.phi
                threshold = 1  # 随机阈值
                self.board_list.append(Board(x_pos, y_pos, threshold))

    def locate_board(self, x, y):
        """
        定位探测器
        :param x: 探测点 x 坐标
        :param y: 探测点 y 坐标
        :return: 在范围内的探测器
        """

        rotation_matrix = np.array([
            [np.cos(self.phi), -np.sin(self.phi)],
            [np.sin(self.phi), np.cos(self.phi)],
        ])
        rotated_coords = np.dot(rotation_matrix, np.array([x, y]))
        local_x, local_y = rotated_coords
        located_board = None
        for board in self.board_list:
            if np.abs(local_x) < board.size_x and np.abs(local_y) < board.size_y:
                located_board = board
                return located_board
        return None

    def eff_Layer(self, x, y):
        """
        计算探测器的几何效率。
        :param x: 探测点 x 坐标
        :param y: 探测点 y 坐标
        :return: 几何效率值
        """
        located_board: Board = self.locate_board(x, y)

        if not located_board:
            return 0
        geo_efficiency = located_board.get_efficiency(x, y)
        return geo_efficiency


class Detector:
    def __init__(self):
        """
        初始化多层探测器，包含多个不同 Z 坐标的 Layer
        """
        self.layers: List[Layer] = []
        self.generator = None
        self.intersections = None

    def add_generator(self, generator):
        """
        添加一个粒子发生器。
        :param generator: 粒子发生器
        """
        self.generator = generator

    def add_layer(self, z_pos, rows, cols, phi=30):
        """
        添加一层探测器。
        :param z_pos: 层的位置（z轴）
        :param rows: 行数
        :param cols: 列数
        :param spacing: 板之间的间距
        """

        layer = Layer(z_pos, phi)
        layer.arrange_boards(rows, cols)
        self.layers.append(layer)

    def find_intersections(self, x0, y0, z0, theta, phi):
        """
        根据发射点和角度，找到射线与所有层的交点。
        :param x0: 发射点的x坐标
        :param y0: 发射点的y坐标
        :param z0: 发射点的z坐标
        :param theta: 水平方向角度
        :param phi: 俯仰方向角度
        :return: 所有穿过的层及交点坐标的列表
        """
        # 计算射线的方向
        dir_x = np.cos(np.radians(theta))
        dir_y = np.sin(np.radians(theta))
        dir_z = np.tan(np.radians(phi))

        intersections = []  # 用于存储所有交点的信息

        for no_layer, layer in enumerate(self.layers):
            if z0 == layer.z_pos:
                continue  # 跳过已经在该层的情况

            # 计算射线与层的交点
            t = (layer.z_pos - z0) / dir_z
            if t <= 0:
                continue  # 如果交点在射线的反方向，跳过

            # 计算交点的 x 和 y 坐标
            x_intersection = x0 + t * dir_x
            y_intersection = y0 + t * dir_y

            # 判断射线是否与层内的板相交
            board = layer.locate_board(x_intersection, y_intersection)
            if board:
                # 保存交点信息
                intersections.append((x_intersection, y_intersection, layer.z_pos, no_layer))
            else:
                intersections.append(None)

        self.intersections = intersections
        return intersections

    def caculate_eff(self):
        """
        计算探测效率。
        """
        efficiencys = []
        for intersection in self.intersections:
            if intersection:
                x, y, z, no_layer = intersection

                efficiencys = self.layers[no_layer].eff_Layer(x, y)
            else:
                efficiency = 0
                print(f"Efficiency at ({x}, {y}, {z}): {efficiency}")
        inefficiency_total = 1

        for efficiency in efficiencys:
            inefficiency_total *= 1 - efficiency
        
        efficiency_total = 1 - inefficiency_total


detector = Detector()

# 2. 添加两层探测器e
detector.add_layer(z_pos=200, rows=2, cols=4, phi=0)  # 第二层在 z=200
detector.add_layer(z_pos=100, rows=2, cols=4, phi=90)  # 第二层在 z=200


# 3. 获取发射点 (0, 0, 0) 以 45 度角发射的探测效率
# 假设发射角度为 theta = 45°（水平），phi = 30°（俯仰角）
efficiency = detector.get_efficiency_at(x=0, y=0, z=0)  # 计算发射点 (0, 0, 0) 的探测效率
print(f"Efficiency at (0, 0, 0): {efficiency}")

# 4. 计算其他位置的效率，例如 (250, 300, 150)
efficiency_2 = detector.get_efficiency_at(x=250, y=300, z=150)
print(f"Efficiency at (250, 300, 150): {efficiency_2}")
