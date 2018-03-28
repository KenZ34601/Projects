# function to check for saddle point
import numpy as np
import math
#from scipy.optimize import linprog
from operator import add, neg

# The following code runs on python version 2.6.7
# 
def saddlecheck(a):
    #maximum, minimum, col,row, s = 0
    row = 0
    col = 0
    s = 0
    nrows = a.shape[0]
    ncols = a.shape[1]
    p = [0]*nrows
    q = [0]*ncols
    v = 0
    for i in range(nrows):
        minimum = a[i][0]
        for j in range(ncols):
            if a[i][j] < minimum:
                minimum = a[i][j]
                col = j; #saving column position of the minimum element of a row
        maximum = a[0][col]
        for k in range(nrows):
            if a[k][col] > maximum:
                maximum = a[k][col]
                row = k; #saving row possition of the maximum element of a row
        if maximum == minimum:
            p[row] = 1
            q[col] = 1
            s = 1
            v = maximum
    if s == 1:
        #print('Value of the Game is', saddle)
        return p,q,v,True
    else:
        return p,q,v,False


def solve_pure_strategy_rowplayer(a):
    n = len(a)
    q = [0] * n
    p = 1
    v = min(a)
    q_index = [i for i, x in enumerate(a) if x == v]
    q[q_index[0]] = 1
    return p, q, v

def solve_pure_strategy_colplayer(a):
    n = len(a)
    p = [0] * n
    q = 1
    v = max(a)
    p_index = [i for i, x in enumerate(a) if x == v]
    p[p_index[0]] = 1
    return p, q, v

# simple 2x2 case
def solve_2by2(a):
    p = (a[1][1]-a[1][0])/((a[0][0] + a[1][1]) - (a[0][1] + a[1][0]))
    q = (a[1][1]-a[0][1])/((a[0][0] + a[1][1]) - (a[0][1] + a[1][0]))
    v = p * a[0][0] + (1-p) * a[1][0]
    p1 = [p, 1-p]
    q1 = [q, 1-q] 
    return p1, q1, v   
    #print('Player I strategy is',[p, 1-p])
    #print('Player II strategy is',[q, 1-q])
    #print('Value of the Game is',v)

def solve_mbyn(a):
    a = a.tolist()
    n = len(a) 
    m = min(a[0] + [0]) #used to make value > 0
    
    T = [[j-m+1 for j in a[i]]+[0]*i+[1]+[0]*(n-i-1)+[1] for i in range(n)]
    T += [[-1]*len(a[0]) + [0]*(n+1)] #initial table
 
    while min(T[-1]) < 0:
        c = T[-1].index(min(T[-1])) #pivot column

        ratios = [i[-1]/float(i[c]) if i[c] > 0 else 'inf' for i in T[:-1]]

        r = ratios.index(min(filter(lambda x: x != 'inf', ratios))) #pivot row
        
        T[r] = [i/float(T[r][c]) for i in T[r]] 
        for i in range(len(T)):
            if i != r:
                T[i] = [T[i][j] - T[i][c]*T[r][j] for j in range(len(T[i]))] #pivot
                for j in range(len(T[i])):
                    if abs(T[i][j]) < pow(10,-8): T[i][j] = 0 #roundoff errors

    v = 1/float(T[-1][-1])

    row = [v*i for i in T[-1][-n-1:-1]]

    col = [0 for i in a[0]]
    for i in range(n):
         c = list(map(list,list(zip(*T)))).index([0]*i+[1]+[0]*(n-i))
         if c < len(col): col[c] += v*T[i][-1]
    v = round(v+m-1,6)
    #print('Player I strategy is', row)
    #print('Player II strategy is',col)
    #print('Value of the Game is',v)
    return row,col,v

def findstrategy(a):
    if a.shape[1] == 1:
        print('There is a saddle point.')
        print('Player I will adopt pure strategy.')
        return solve_pure_strategy_colplayer(a)
    if a.shape[1] == None:
        print('There is a saddle point.')
        print('Player II will adopt pure strategy.')
        return solve_pure_strategy_rowplayer(a)
    if len(a) ==2 and len(a[1,:]) == 2:
        p,q,v,string = saddlecheck(a)
        if string == False:
            print('There is no saddle point.')
            return solve_2by2(a)
        else:
            print('There is no saddle point.')
            print('Player I and II will adopt pure strategy.')
            return p,q,v
    else:
        p,q,v,string = saddlecheck(a)
        if string == False:
            print('There is no saddle point.')
            return solve_mbyn(a)
        else:
            print('There is a saddle point.')
            print('Player I and II will adopt pure strategy.')
            return p,q,v

