# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/4  21:06
# @Author: Yanjun Hao
# @File  : ds.py

import random
import math
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from rich import print
import logging
import time
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT_0 = FILE.parents[0]  # project root directory
ROOT_1 = FILE.parents[1]  # project root directory
if str(ROOT_0) not in sys.path:
    sys.path.append(str(ROOT_0))  # add ROOT to PATH
if str(ROOT_1) not in sys.path:
    sys.path.append(str(ROOT_1))  # add ROOT to PATH


class MatrixSolving:
    def __init__(self, data: list):
        self.xi_li = data
        self.n = len(self.xi_li)
        self.r0 = random.uniform(0, min(self.xi_li))
        self.F_xi_li = self.get_Fx_i()
        self.Xi_li = self.get_Xi()
        self.Yi_li = self.get_Yi()
        self.b0, self.th0 = self.get_b0_th()
        self.check()
        self.E1_E9_li = self.get_E1_E9()
        self.R0 = 0.5

    def get_Fx_i(self) -> list:
        Fx_i_list = []
        for idx in range(self.n):
            Fx_i = (idx + 1 - 0.3) / (self.n + 0.4)
            Fx_i_list.append(Fx_i)
        return Fx_i_list

    def get_Xi(self) -> list:
        return list(map(lambda x: math.log(x - self.r0), self.xi_li))

    def get_Yi(self) -> list:
        return list(map(lambda x: math.log(math.log(1 / (1 - x))), self.F_xi_li))

    def linear_regression(self) -> object:
        model = LinearRegression()
        model.fit(np.array(self.Xi_li).reshape((-1, 1)), np.array(self.Yi_li))
        return model.coef_[0], model.intercept_

    def get_b0_th(self):
        coef_a, coef_b = self.linear_regression()
        # print(f"coef_a={coef_a}, coef_b={coef_b}")
        real_b0 = coef_a
        real_th0 = (math.e ** (-coef_b / real_b0)) + self.r0
        return real_b0, real_th0

    def get_E1_E9(self) -> list:
        E1 = sum(list(map(lambda x: 1 / (x - self.r0), self.xi_li)))
        E2 = sum(list(map(lambda x: (x - self.r0) ** self.b0, self.xi_li)))
        E3 = sum(list(map(lambda x: (x - self.r0) ** (self.b0 - 1), self.xi_li)))
        E4 = sum(list(map(lambda x: math.log(x - self.r0), self.xi_li)))
        E5 = sum(list(map(lambda x: ((x - self.r0) ** self.b0) * (math.log(x - self.r0)), self.xi_li)))
        E6 = sum(list(map(lambda x: 1 / ((x - self.r0) ** 2), self.xi_li)))
        E7 = sum(list(map(lambda x: (x - self.r0) ** (self.b0 + 2), self.xi_li)))
        E8 = sum(list(map(lambda x: ((x - self.r0) ** (self.b0 - 1)) * (math.log(x - self.r0)), self.xi_li)))
        E9 = sum(list(map(lambda x: ((x - self.r0) ** self.b0) * ((math.log(x - self.r0)) ** 2), self.xi_li)))
        return [E1, E2, E3, E4, E5, E6, E7, E8, E9]

    def get_A_B(self) -> np.array:
        # print('哈哈', (self.th0 - self.r0) ** (self.b0 + 1))
        # print('哈哈', self.th0, self.r0, self.b0)
        a12 = ((self.n * self.b0) / ((self.th0 - self.r0) ** 2)) - \
              ((self.b0 - 1) * self.E1_E9_li[5]) - \
              ((self.b0 * (self.b0 + 1) * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** (self.b0 + 1))) + \
              ((2 * self.b0 * self.E1_E9_li[2]) / ((self.th0 - self.r0) ** (self.b0 + 1))) - \
              ((self.b0 * (self.b0 - 1) * self.E1_E9_li[6]) / ((self.th0 - self.r0) ** self.b0))
        a13 = (self.n / (self.th0 - self.r0)) - self.E1_E9_li[0] + \
              (((self.b0 * math.log(self.th0 - self.r0) - 1) * self.E1_E9_li[1]) / (
                      (self.th0 - self.r0) ** (self.b0 + 1))) - \
              ((self.b0 * self.E1_E9_li[4]) / ((self.th0 - self.r0) ** (self.b0 + 1))) - \
              (((self.b0 * math.log(self.th0 - self.r0) - 1) * self.E1_E9_li[2]) / ((self.th0 - self.r0) ** self.b0)) - \
              ((self.b0 * self.E1_E9_li[7]) / ((self.th0 - self.r0) ** self.b0))
        a11 = (self.n * self.b0) / ((self.th0 - self.r0) ** 2) + \
              ((self.b0 * (self.b0 + 1) * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** (self.b0 + 1))) + \
              (((self.b0 ** 2) ** self.E1_E9_li[2]) / ((self.th0 - self.r0) ** self.b0))
        a22 = (-self.n / self.b0) - \
              ((((math.log(self.th0 - self.r0)) ** 2) * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** self.b0)) + \
              ((2 * (math.log(self.th0 - self.r0)) * self.E1_E9_li[4]) / ((self.th0 - self.r0) ** self.b0)) - \
              (self.E1_E9_li[8] / ((self.th0 - self.r0) ** self.b0))
        a23 = (self.n / self.b0) - \
              (((self.b0 * math.log(self.th0 - self.r0) - 1) * self.E1_E9_li[1]) / (
                      (self.th0 - self.r0) ** (self.b0 + 1))) + \
              ((self.b0 * self.E1_E9_li[4]) / ((self.th0 - self.r0) ** (self.b0 + 1)))
        a33 = ((self.n * self.b0) / ((self.th0 - self.r0) ** 2)) - \
              ((self.b0 * (self.b0 + 1) * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** (self.b0 + 2)))
        b1 = ((-self.n * self.b0) / (self.th0 - self.r0)) + \
             ((self.b0 - 1) * self.E1_E9_li[0]) + \
             ((self.b0 * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** (self.b0 + 1))) - \
             ((self.b0 * self.E1_E9_li[2]) / ((self.th0 - self.r0) ** self.b0))
        b2 = (self.n / self.b0) + \
             (self.n * math.log(self.th0 - self.r0)) + \
             self.E1_E9_li[3] - \
             ((math.log(self.th0 - self.r0) * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** self.b0)) - \
             (self.E1_E9_li[4] / ((self.th0 - self.r0) ** self.b0))
        b3 = ((self.n * self.b0) / (self.th0 - self.r0)) - \
             ((self.b0 * self.E1_E9_li[1]) / ((self.th0 - self.r0) ** (self.b0 + 1)))
        a21 = a12
        a31 = a13
        a32 = a23
        return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]]), np.array([[b1], [b2], [b3]])

    def get_X(self) -> np.array:
        A, B = self.get_A_B()
        X = np.dot(np.linalg.inv(A), B)
        # print('-' * 50)
        # print(A, B, X)
        # print('-' * 50)
        return X

    def update_coef(self):
        while True:
            # print(f"r0={self.r0}, b0={self.b0}, th0={self.th0}", (self.th0 - self.r0) ** (self.b0 + 1))
            X_coef = self.get_X()
            R1 = abs(X_coef[0, 0] / self.r0) + abs(X_coef[1, 0] / self.b0) + abs(X_coef[2, 0] / self.th0)
            if R1 > self.R0:
                print("发散")
                self.r0 = random.uniform(0, min(self.xi_li))  # 重新选择r0
                self.Xi_li = self.get_Xi()  # 重新计算Xi
                self.R0 = 0.5
                self.b0, self.th0 = self.get_b0_th()
                print(f"r0={self.r0}, b0={self.b0}, th0={self.th0}")
                self.check()
                self.E1_E9_li = self.get_E1_E9()
            else:
                if R1 < 0.001:
                    print("程序运行结束")
                    print(f"[italic bold red]r0: {self.r0}, b0: {self.b0}, th0: {self.th0}")
                    with open("../run/result.json", "w") as f:
                        f.write(json.dumps({"r0": self.r0, "th0": self.th0, "b0":self.b0}, indent=4))
                    break
                else:
                    self.R0 = R1
                    self.r0 = self.r0 + X_coef[0, 0]
                    self.b0 = self.b0 + X_coef[1, 0]
                    self.th0 = self.th0 + X_coef[2, 0]
                    self.check()
                    self.E1_E9_li = self.get_E1_E9()

    def check(self):
        # NOTE: --------------------------- 判断self.th0 - self.r0的值 ----------------------------------
        # (abs(self.th0 - self.r0) < 0.01) or
        while (self.th0 < self.r0) or (math.isinf(self.th0)) or ((self.th0 - self.r0) < 1e-5) or \
                (self.r0 >= min(self.xi_li)):
            self.r0 = random.uniform(0, min(self.xi_li))  # 重新选择r0
            self.Xi_li = self.get_Xi()  # 重新计算Xi
            self.R0 = 0.5
            self.b0, self.th0 = self.get_b0_th()
        # NOTE: --------------------------- 判断self.th0 - self.r0的值 ----------------------------------


if __name__ == '__main__':
    input_data = [60.46, 58.72, 61.25, 55.73, 60.70, 60.71, 60.82, 61.21, 55.13, 59.14, 56.13, 62.44, 64.29, 61.22,
                  59.14]
    matrix_solving = MatrixSolving(data=input_data)
    matrix_solving.update_coef()
