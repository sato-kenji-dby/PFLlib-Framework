#!/usr/bin/env python
# coding: utf-8

import numpy as np
from math import floor

def fun_p(d,S,G):
    '''Global P is calculated here'''
    return ...


def fun_C(strategy_list,i,M):
    '''Invididual rewards calculated here'''
    return ...

def Potential(strategy_list,M):
    '''Global Potential is calculated here'''
    return ...

def vio_approach(stra_len, user_strategy_list, M, mode = "DBR"):
    global num
    Utility = np.zeros([1,num])
    update_list = np.zeros([1,num])
    temp_user_list = user_strategy_list.copy()
    for i in range(num):
        C = fun_C(user_strategy_list,i,M)
        Utility[0][i] = C

        if mode == "GCA":
            lb = [1e-3] * 1
            ub = [1] * 1
        else:
            lb = [1e-3] * 2
            ub = [1] * 2
        bnds = tuple([(lb[i], ub[i]) for i in range(len(lb))])

        '''solving with GBD algorithm'''

        if -Cmax > C:
            if mode == "GCA":
                d = df
                f = np.clip(d * 1, 0, 1)
                df = np.array([d, f]).reshape(2)
                #print(i)
            temp_user_list[:, i] = df
            update_list[0][i] = 1

    user_strategy = temp_user_list
    return [user_strategy,update_list,Utility]

def fun_Chan(df, strategy_list, i, M, mode="DBR"):
    sl = np.array(list(strategy_list))
    sl[:, i] = df
    if mode == "GCA":
        d = sl[0]
        f = np.clip(d * 1, 0, 1)
        sl = np.array([d, f])
    c = fun_C(sl,i,M)
    return -c


num = 4
max_round = 100
source = 1
G = 100

[S,F,B,P,ka,tao,yeta,rho,gamma,omega,strategy_pool,stra_len,D_T] = ini_para(num)
M = cooperation(num)
[Z,X,W,h] = depent_para(S,F,P,ka,yeta,rho,omega,M,num)

if source == 1:
    with open("./default.json") as f:
        config = json.load(f)
    S = np.array([config['S']])[:, :num]
    F = np.array([config['F']])[:, :num]
    B = np.array([config['B']])[:, :num]
    P = np.array([config['P']])[:, :num].T
    ka = np.array(config['k'])
    tao = np.array(config['tao'])
    yeta = np.array([config['yeta']])[:, :num]
    gamma = np.array(config['r'])
    D_T = np.array([config['D_T']])[:, :num]
    M = np.array(config['M'])[:num, :num]
    [Z,X,W,h] = depent_para(S,F,P,ka,yeta,rho,omega,M,num)

def DBR():
    user_strategy_list = np.zeros([2,num])
    user_strategy_list[0] = min(strategy_pool[0])
    user_strategy_list[1] = max(strategy_pool[1])
    change_num = 1
    the_round = 0
    Utility = np.zeros([1,num])
    
    overall_strategy = np.zeros([the_round,2,num])
    overall_potential = np.zeros(the_round)
    overall_utility = np.zeros([the_round,num])
    
    U = Potential(user_strategy_list,M)
    while the_round < max_round:
        the_round += 1
    
        [temp_user_list,update_list,Utility] = vio_approach(stra_len,user_strategy_list,M)
        if np.sum(update_list) > 0.5:
            idx = [i for i in range(len(update_list[0])) if update_list[0][i] > 0]
            change_num = min(np.sum(update_list), change_num)
            luck = np.random.choice(idx, int(change_num), replace=False)
            for change in luck:
                user_strategy_list[:, change] = temp_user_list[:, change]
    
            U = Potential(user_strategy_list,M)
    
        overall_strategy = np.vstack((overall_strategy, np.expand_dims(user_strategy_list,axis=0)))
        overall_potential = np.array(list(overall_potential) + [U]) # =U
        overall_utility = np.vstack((overall_utility, Utility))
        if np.sum(update_list) < 0.5:
            print("")
            break
    sumU = [sum(i) for i in overall_utility]
    
    #print(overall_potential)
    #print(sumU)
    #print(user_strategy_list)
    print(sum(user_strategy_list[0]) / num)
    print(sum(user_strategy_list[1]) / num)


def CGBD():
    tr = lambda df: np.array([[df[i] for i in range(len(df)//2)], [df[i] for i in range(len(df)//2, len(df))]])
    fun = lambda df: -Potential(tr(df),M)
    lb = [1e-3] * 2 * num
    ub = [1] * 2 * num
    bnds = tuple([(lb[i], ub[i]) for i in range(len(lb))])
    
    '''solving with GBD algorithm'''
    r = sum([fun_C(tr(df),i,M) for i in range(num)])
    
    # print(Cmax)
    #print(df)
    # print([fun_C(tr(df),i,M).tolist() for i in range(num)])
    # print(r)
    print(sum(df[:num]) / num)
    print(sum(df[num:]) / num)

DBR()
CGBD()
