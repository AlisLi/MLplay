import numpy as np
from .metrics import r2_score

# 简单线性回归算法

class SimpleLinearRegression1:

    def __init__(self):
        """初始化Simple_Linear_Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        """根据训练数据集x_train和y_train来训练Simple_Linear_Regression 模型"""
        assert x_train.ndim == 1, \
            "simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train),\
            "the size of x_train must be equal to the size if y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        #分母
        num = 0.0
        #分子
        d = 0.0
        for x_i,y_i in zip(x_train,y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) ** 2
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self,x_predict):
        """给定待预测的数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给单个待预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:

    def __init__(self):
        """初始化Simple_Linear_Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self,x_train,y_train):
        """根据训练数据集x_train和y_train来训练Simple_Linear_Regression 模型"""
        assert x_train.ndim == 1, \
            "simple Linear Regressor can only solve single feature training data"
        assert len(x_train) == len(y_train),\
            "the size of x_train must be equal to the size if y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        #分母
        num = 0.0
        #分子
        d = 0.0

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self,x_predict):
        """给定待预测的数据集x_predict，返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
            "simple Linear Regressor can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """给单个待预测数据x_single,返回x_single的预测结果值"""
        return self.a_ * x_single + self.b_

    def score(self,x_test,y_test):
        """根据测试数据集x_test和y_test确定当前模型的准确度"""

        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "SimpleLinearRegression2()"