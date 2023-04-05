# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2023/4/5  21:15
# @Author: Yanjun Hao
# @File  : plot.py

import random
import math
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from rich import print
import seaborn as sns
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT_0 = FILE.parents[0]  # project root directory
ROOT_1 = FILE.parents[1]  # project root directory
if str(ROOT_0) not in sys.path:
    sys.path.append(str(ROOT_0))  # add ROOT to PATH
if str(ROOT_1) not in sys.path:
    sys.path.append(str(ROOT_1))  # add ROOT to PATH

import matplotlib
font = {'family': 'Times New Roman', 'size': '18'}  # SimSun宋体 'weight':'bold',
matplotlib.rc('font', **font)


class PlotFigure:
    def __init__(self, data: list):
        self.xi_li = data
        self.n = len(self.xi_li)
        self.r0 = 31.82352735151358
        self.F_xi_li = self.get_Fx_i()
        self.Xi_li = self.get_Xi()
        self.Yi_li = self.get_Yi()
        self.b0, self.th0 = self.get_b0_th()
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
        real_b0 = coef_a
        real_th0 = (math.e ** (-coef_b / real_b0)) + self.r0
        return real_b0, real_th0

    def plot_figure(self):
        model = LinearRegression()
        model.fit(np.array(self.Xi_li).reshape((-1, 1)), np.array(self.Yi_li))
        print(f"coef={model.coef_[0]}, intercept={model.intercept_}")
        with open("../run/liner_regression.txt", "w") as f:
            f.write(f"coef={model.coef_[0]}, intercept={model.intercept_}")
        df = pd.DataFrame({"X": self.Xi_li, "Y": self.Yi_li})
        plt.figure(figsize=(10, 7))
        sns.regplot(x="X", y="Y", data=df, line_kws={'linewidth': 2, 'color': 'red'})
        # plt.xlim(3,4)
        plt.savefig("../run/liner_regression.svg", bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    input_data = [60.46, 58.72, 61.25, 55.73, 60.70, 60.71, 60.82, 61.21, 55.13, 59.14, 56.13, 62.44, 64.29, 61.22,
                  59.14]
    matrix_solving = PlotFigure(data=input_data)
    matrix_solving.plot_figure()
