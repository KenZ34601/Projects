#----------------------------------------------------------------------------#
# Title:          Game Theory HW2                                            #
# Author:         Ken Zhou                                                   #
# Date:           02/28/17                                                   #
# Description:    Problem 1 is about finding G function that given a set of  #
#                 positions X, and a follower function F, there is a max non #
#                 negative interger not in the G(y) expressed as mex(G(y)).  #
#                 For example, for a game G of 21 chips with the subtraction #
#                 set S = {1,2,3}, the game can be represented by:           #
#                 G = {(0, ∅), (1, {0}), (2, {0, 1}), (3, {0, 1, 2}), (4,    #
#                 {1, 2, 3}), ..., (21, {18, 19, 20})}                       #
# Note:           The answer is written in python							 #
#----------------------------------------------------------------------------#
import numpy as np

def mexcompute(F):
    mex = 0
    F.sort()
    for mex in range(len(F)):
        while(F[mex]!=mex):
            return mex
            mex = mex+1
    return mex+1

def Gfunction(n):
    #F_pos = np.array([], dtype=np.float64)
    F_pos = []
    i=1
    if n==0:
        return 0
    if n==1:
        return 1
    if n==2:
        return 2
    if n==3:
        return 3
    else:
        while(i <= 3):
            F_pos.append(Gfunction(n - i))
            #F_pos = Gfunction(n - i)
            #print(F_pos)
            i = i + 1
        #print(F_pos)
        return mexcompute(F_pos)

#n = 100
#print('G value of 100 chips is', Gfunction(n)) 
