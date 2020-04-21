#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.special import comb
import itertools
import numpy as np
import random


# In[2]:


# 交叉
class CrossOver:
    def __init__(self):
        pass
    
    def cross_over(self, x1, x2):
        pass
    
# 一点交叉
class OnePointCrossOver(CrossOver):
    def __init__(self, length, point):
        self.length = length
        self.point = point
        if point <= 0 and point >= length:
            print("パラメータの設定に問題があります")
    
    def cross_over(self, x1, x2):
        y1 = np.hstack((x1[0 : self.point], x2[self.point : self.length]))
        y2 = np.hstack((x2[0 : self.point], x1[self.point : self.length]))
        return y1, y2
    
# 一様交叉
class UniformCrossOver(CrossOver):
    def __init__(self, length, mask):
        self.length = length
        self.mask = mask
        if self.lenght != len(self.mask):
            print("パラメータの設定に問題があります")
    
    def cross_over(self, x1, x2):
        y1 = []
        y2 = []
        for i in range(self.length):
            y1.append(x1[i] if self.mask[i] else x2[i])
            y2.append(x2[i] if self.mask[i] else x1[i])
        return y1, y2


# In[3]:


class Mutation:
    def __init__(self, f, mutation):
        self.f = f
        self.mutation = mutation
        
    def mutation_func(self, x):
        return self.f(x, self.mutation)


# In[4]:


class MOEAD:
    # m … 目的数
    # H … 分解パラメータ
    # N … 重みベクトルと解集団サイズ
    # T … 近傍サイズ
    # f … 関数集団
    # cross_over … 交叉方法
    # mutation … 突然変異
    def __init__(self, m, H, T, fs, cross_over, mutation):
        self.m = m
        self.H = H
        self.N = comb(self.H + self.m - 1, self.m - 1, exact=True)
        self.T = T
        self.fs = fs
        self.cross_over = cross_over
        self.mutation = mutation
        print("m:%d, H:%d, N:%d, T:%d" % (self.m, self.H, self.N, self.T))
        self.init_x_func = None
        self.init_z_func = None
        
    # 初期化フェーズ
    def init_phase(self):
        # 重みベクトル群の生成
        self.L = []
        for combo in itertools.combinations(range(self.H + self.m - 1), self.m - 1):
            lamb = [combo[0]]
            for i in range(1, self.m - 1):
                lamb.append(combo[i] - combo[i - 1] - 1)
            lamb.append(self.H + self.m - 2 - combo[self.m - 2])
            
            self.L.append(np.array(list(map(lambda x: x / self.H  + 0.0001, lamb))))
        if len(self.L) != self.N:
            print("重みベクトルの生成数が正しくありません N:%d != L:%d" % (self.N, self.L))
            
        # 近傍重みベクトル群を見つける
        self.B = [None for _ in range(len(self.L))]
        for i, lamb_i in enumerate(self.L):
            distances = {idx : np.linalg.norm(lamb_i - lamb_j) for idx, lamb_j in enumerate(self.L)}
            sorted_dis = sorted(distances.items(), key = lambda x : x[1])
            self.B[i] = sorted([sorted_dis[k][0] for k in range(self.T)])
            
        # 初期集団生成
        if self.init_x_func == None:
            init_rand_min = 1.0
            init_rand_max = 2.0
            self.x = np.array([lamb * random.uniform(init_rand_min, init_rand_max) for lamb in self.L])
        else:
            self.x = self.init_x_func(self.L)
        
        # 理想点初期化
        if self.init_z_func == None:
            self.z = np.array([0 for _ in range(self.m)])
        else:
            self.z = self.init_z_func(self.m)
        
            
    def solution_search_phase(self, generation):
        for g in range(generation):
            for i in range(self.N):
               # 親選択
                p, q = random.sample(self.B[i], 2)
                x_p, x_q = self.x[p], self.x[q]
                
                #交叉
                y1, y2 = self.cross_over.cross_over(x_p, x_q)
                y = y1 if random.randrange(2) == 0 else y2
                #突然変異
                self.mutation.mutation_func(y)
                    
                profit_y = [f(y) for f in self.fs]
                
                # 理想点更新
                for j in range(self.m):
                    if profit_y[j] < self.z[j]:
                        self.z[j] = y[j]
                
                # 解集団の更新
                for j in self.B[i]:
                    lamb = self.L[j]
                    g_x = max([f(self.x[j]) / lamb[k] for k, f in enumerate(self.fs)])
                    g_y = max([fy / lamb[k] for k, fy in enumerate(profit_y)])
                    if g_y < g_x:
                        self.x[j] = y
        return self.x        
    
    def set_init_x_func(self, init_x_func):
        self.init_x_func = init_x_func
        
    def set_init_z_func(self, init_z_func):
        self.init_z_func = init_z_func


# In[ ]:




