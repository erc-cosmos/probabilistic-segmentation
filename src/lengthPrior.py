#!/usr/bin/python3

import scipy as sp
import scipy.stats

class NormalLengthPrior:
    def __init__(self,mean,stddev,x,maxLength):
        self.mean = mean
        self.stddev = stddev
        self.x = x
        self.max = maxLength

    def evalCond(self,N,i,j):
        """ Returns the likelihood of a boundary at j given a boundary at i
        N -- not used anymore TODO: Remove
        i -- known boundary location
        j -- examined location
        """
        if j == i or j>=len(self.x):
            return 0
        x0 = self.x[i]
        xmax = min(self.x[-1]-x0, self.max)
        xjpred = self.x[j-1]-x0
        xj = self.x[j]-x0

        if xj > self.max: # Cap xj at max
            xj = self.max
        if xjpred > self.max: # Cap xj at max
            xjpred = self.max
        
        c0,cjpred,cj,cmax = sp.stats.norm.cdf([0,xjpred,xj,xmax],self.mean,self.stddev)
        
        scaling = cmax - c0 # Scaling factor for the part of the distribution outside possible values
        if scaling == 0:
            print('ohoho')
            return 1
        return (cj-cjpred)/scaling
    
    def getMaxIndex(self,i):
        """ Return the maximum arc end with non-null prior for an arc starting at i."""
        xmax = self.x[i] + self.max
        imax = i + next((it for it,x in enumerate(self.x[i:]) if x>xmax),
                        len(self.x[i:])-1)
        return imax
    
    def getMinIndex(self,j):
        """ Return the minimum arc start with non-null prior for an arc ending at i."""
        xmin = self.x[j] - self.max
        imin = next((it-1 for it,x in enumerate(self.x[:j]) if x>xmin),0)
        if imin == -1:
            imin = 0
        return imin