# -*- coding: utf-8 -*-
"""
Created on Sun Jan 7 18:50:54 2018

@author: bitcit
"""

import numpy as np

class svdModel(object):
    '''
    SVD++模型
    @parameter
        data : 输入的训练数据，为2维array,第0列为user_id, 第1列为item_id, 第2列为rate
        dimension : 确定SVD分解的维度，即 p_u, q_i, y_i, b_i, b_u 的维度
        param_lambda : 正则项系数 lambda
        learning_rate : 进行梯度下降更新时的学习率，也叫步长
        iteration : 梯度下降时迭代的次数
    
    @attribute
        self.data :
        self.dimension :
        self.param_lambda :
        self.learning_rate :
        self.iteration : 
        self.userNum : int类型，训练数据中user的数量
        self.itemNum : intl类型，训练数据中item的数量
        self.userMap : dict类型，将原始的user_id一对一映射到0~self.userNum-1之间
        self.itemMap : dict类型，将原始的item_id一对一映射到0~self.itemNum-1之间
        self.minRate : 训练数据中rate的最小值
        self.maxRate : 训练数据中rate的最大值
        self.mean : 训练数据中rate的平均值
        self.rateMatrix : 2d-array, shape = self.userNum * self.itemNum 由训练数据得到的评分矩阵
        self.p : 2d-array, shape = self.userNum * self.dimension, SVD++模型中的矩阵P
        self.q : 2d-array, shape = self.itemNum * self.dimension, SVD++模型中的矩阵Q
        self.bu : 1d-array, shape = self.userNum , SVD模型中的user bias
        self.bi : 1d-array, shape = self.itemNum , SVD模型中的item bias

        
    @function
        fit() : 模型训练，和sklearn中模型训练一样的fit()函数类似
        predict(test_data=None) : 用训练好的模型进行预测，和sklearn中模型预测的predict()函数类似
    
    '''
    def __init__(self, data=None, dimension=2, param_lambda=0.0, learning_rate=0.1, iteration=100):
        self.dimension = dimension
        self.param_lambda = param_lambda
        self.learning_rate = learning_rate
        self.iteration = iteration
        if(data is None):
            raise Exception("@parameter 'data' can not be None!" )
        self.data = data
        self.userMap = {}
        self.itemMap = {}
        self._formRateMatrix()
        self.mean = 0.0
        self.maxRate = 0
        self.minRate = 0
        self.p = np.random.random((self.userNum, self.dimension)) /10     
        self.bu = np.zeros(self.userNum, dtype=float)
        self.q = np.random.random((self.itemNum, self.dimension)) /10
        self.bi = np.zeros(self.itemNum, dtype=float)
        self.train_error = []
        

    def _formRateMatrix(self):
        self.userNum = 0
        self.itemNum = 0
        for i in range(len(self.data)):
            user = self.data[i][0]
            item = self.data[i][1]
            if not(user in self.userMap):
                self.userMap[user] = self.userNum
                self.userNum += 1
            if not (item in self.itemMap):
                self.itemMap[item] = self.itemNum
                self.itemNum += 1
                
        self.rateMatrix = np.zeros((self.userNum, self.itemNum), dtype=float)
        for i in range(len(self.data)):
            user = self.data[i][0]
            item = self.data[i][1]
            rate = self.data[i][2]
            if i==0:
                self.minRate = rate
                self.maxRate = rate
            if rate > self.maxRate:
                self.maxRate = rate
            if rate < self.minRate:
                self.minRate = rate
            self.rateMatrix[self.userMap[user]][self.itemMap[item]] = rate
        self.mean = np.sum(self.rateMatrix)/len(self.data)
        print("min: ", self.minRate)
        print("max: ", self.maxRate)
         

    def fit(self):
        print("####### start traning #######")
        self._gradientDescent()
        
        
        
    
    def _gradientDescent(self):
        for ite in range(self.iteration):
            RMSE = 0.0
            MAE = 0.0
            Rate_num = 0
            for u in range(self.userNum):
                for i in range(self.itemNum):
                    
                    if np.fabs(self.rateMatrix[u][i]) < 1e-6:
                        continue                      
                    rui = self.mean + self.bu[u] + self.bi[i] + self.p[u].dot(self.q[i])
                    '''
                    if rui > self.maxRate:
                        rui = self.maxRate
                    if rui < self.minRate:
                        rui = self.minRate
                    '''    
                    error = self.rateMatrix[u][i] - rui
                    '''
                    if(i == 5):
                        raise Exception("111")
                        '''
                    #print("error:", error)
                    
                    delta_bu, delta_bi, delta_p, delta_q = self._calGradient(error, u, i)
                    self.bu[u] = self.bu[u] - self.learning_rate * delta_bu
                    self.bi[i] = self.bi[i] - self.learning_rate * delta_bi
                    self.p[u] = self.p[u] - self.learning_rate * delta_p
                    self.q[i] = self.q[i]- self.learning_rate * delta_q
                    
            for u in range(self.userNum):
                for i in range(self.itemNum):
                    if np.abs(self.rateMatrix[u][i]) <1e-6:
                        continue
                    rui = self.mean + self.bu[u] + self.bi[i] + self.p[u].dot(self.q[i])
                    error = self.rateMatrix[u][i] - rui
                    Rate_num += 1
                    RMSE += error*error
                    MAE += np.abs(error)
                    
                        
                    
            self.train_error.append(RMSE/Rate_num)
            self.learning_rate *= 0.9
            print("Iteration {}: RMSE = {:.6f}, MAE = {:.6f}".format(ite, RMSE/Rate_num, MAE/Rate_num))
            
    def _calGradient(self, error, u, i):
        delta_bu = -error + self.param_lambda*self.bu[u]
        delta_bi = -error + self.param_lambda*self.bi[i]
        delta_p = -error*self.q[i] + self.param_lambda*self.p[u]
        delta_q = -error*self.p[u] + self.param_lambda*self.q[i]
        return delta_bu, delta_bi, delta_p, delta_q

    
    def _calMAE(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))                    
            
    def predict(self, test_data=None):
        if test_data is None:
            raise Exception("@parameter test_data can not be None")
        print("####### start predict ########")
        rate = np.zeros(len(test_data))
        unhitted = len(test_data)
        for index in range(len(test_data)):
            original_user = test_data[index][0]
            original_item = test_data[index][1]
            rate[index] = self.mean
            if original_user in self.userMap:
                user = self.userMap[test_data[index][0]]
                rate[index] += self.bu[user]
            if original_item in self.itemMap:
                item = self.itemMap[test_data[index][1]]
                rate[index] += self.bi[item]
            if (original_user in self.userMap) and (original_item in self.itemMap):
                rate[index] += self.p[user].dot(self.q[item])
                unhitted -= 1
                           
        print("test MAE: ", self._calMAE(test_data[:,2], rate))
        print("test RMSE: ", np.sqrt(np.mean((test_data[:, 2] - rate)**2)))
        print("unhitted: ", unhitted)
        return rate
        

if __name__ == "__main__":

    data = np.loadtxt("ua.base", dtype=int)
    model = svdModel(data=data, dimension=10, param_lambda=0.1,learning_rate=0.1, iteration=50)
    model.fit()
    test = np.loadtxt("ua.test", dtype=int)
    model.predict(test_data=test)
    
    
    
    
        
        
        
