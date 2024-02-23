import numba
import mlframe
import numpy as np
import pandas as pd
from numba import cuda,njit,prange

################################################################################################
#ARRAY STATS
################################################################################################

@njit(fastmath=True)
def arrayMinMax(x,l=0,r=0):
    if r==0: r=len(x)
    firstElem=x[l]
    maximum,minimum=firstElem,firstElem
    for v in x[l:r]:
        if v > maximum:
            maximum =v
        elif v < minimum:
            minimum =v
    return (minimum, maximum)
@njit(fastmath=True,parallel=True)
def arrayMinMaxParallel(array,l=0,r=0,maxThreads=8):
    arrLen=len(array)
    if r==0: r=arrLen
    nElemsToProcess=(r-l)
    nThreads=min(max(nElemsToProcess,1),maxThreads)            
    chunkSize=nElemsToProcess//nThreads
    minMaxData=np.empty((nThreads,2),array.dtype)
    for k in prange(nThreads):
        lBound=l+chunkSize*k;rBound=l+chunkSize*(k+1)
        if k==nThreads-1:
            rBound=r
        minMaxData[k,:]=arrayMinMax(array,lBound,rBound)
    return np.min(minMaxData[:,0]),np.max(minMaxData[:,1])
@njit(fastmath=True,parallel=True)
def npnbArrayMinMax(x):
    return x.min(), x.max()
################################################################################################
#ARRAY SORTING
################################################################################################
@njit(fastmath=True)
def arrayCountingSort(array,maxval):    
    res=np.empty(len(array),np.int32)
    m = maxval + 1
    count=np.zeros(m,np.int32)
    for a in array:
        count[a] += 1             # count occurences
    i = 0
    for a in range(m):            # emit
        for c in range(count[a]): # - emit 'count[a]' copies of 'a'
            res[i] = a
            i += 1
    return res
################################################################################################
#ARRAY ARGSORTING
################################################################################################
@njit(fastmath=True)
def emptyListOfInts():
    return [i for i in range(0)]
@njit(fastmath=True)
def BinByUniqueValues(array,l,r,m,mask):        
    groupedIndices=[emptyListOfInts() for k in range(m)]
    if len(mask)>0:
        i=l
        while i<r:        
            ind=mask[i]
            groupedIndices[array[ind]].append(ind)
            i+=1
    else:
        i=l
        while i<r:
            groupedIndices[array[i]].append(i)
            i+=1
    #print("l=",l,"r=",r) #,groupedIndices,'\n'
    return groupedIndices
    #cGrowthFactor=2
    #if v>m:            
    #    newM=m*cGrowthFactor
    #    #print ("resizing from %d to %d" %(m,newM))
    #    count+=[[i for i in range(0)] for k in range(newM-m)]
    #   m=newM
@njit(fastmath=True)
def arrayCountingArgSort(array,maxval,mask=np.array([],np.int32)):        
    m=maxval+1    
    
    #Allocate output array
    if len(mask)>0:
        arrLen=len(mask)
    else:
        arrLen=len(array)        
    argsorted=np.empty(arrLen,np.int32)    
    
    #Group indices of same values
    groupedIndices=BinByUniqueValues(array,0,arrLen,m,mask)
    
    position = 0
    for k in range(m):
        if len(groupedIndices[k])>0:
            for index in groupedIndices[k]:
                argsorted[position] = index
                position+= 1            
    return argsorted    
@njit(fastmath=True)
def arrayCountingArgSortAndUniqueValues(array,maxval,mask=np.array([],np.int32)):        
    m=maxval+1    
    
    #Allocate output array
    if len(mask)>0:
        arrLen=len(mask)
    else:
        arrLen=len(array)        
    argsorted=np.empty(arrLen,np.int32)    
    
    #Group indices of same values
    groupedIndices=BinByUniqueValues(array,0,arrLen,m,mask)
    
    position = 0
    uniqueValues=emptyListOfInts()
    uniqueValuesIndices=emptyListOfInts()
    for k in range(m):
        if len(groupedIndices[k])>0:
            uniqueValues.append(k)
            uniqueValuesIndices.append(position)
            for index in groupedIndices[k]:
                argsorted[position] = index
                position+= 1            
    return np.array(uniqueValues,np.int32),np.array(uniqueValuesIndices,np.int32),argsorted
@njit(fastmath=True,parallel=True)
def arrayCountingArgSortThreaded(array,maxval,mask=np.array([],np.int32),maxThreads=2):
    m=maxval+1    
    
    #Allocate output array
    if len(mask)>0:
        arrayLen=len(mask)
    else:
        arrayLen=len(array)
    argsorted=np.empty(arrayLen,np.int32)
    
    #Group indices of same values    
    effectiveSize=int(m*3)        
    if arrayLen<=effectiveSize: 
        nThreads=1
    else:
        nThreads=min(max(arrayLen//effectiveSize,1),maxThreads)            
    groups=[[emptyListOfInts() for k in range(0)]]*nThreads
    chunkSize=arrayLen//nThreads
    #print("nThreads=",nThreads)
    for k in prange(nThreads):
        lBound=chunkSize*k;rBound=chunkSize*(k+1)
        if k==nThreads-1:
            rBound=arrayLen
        groups[k]=BinByUniqueValues(array,lBound,rBound,m,mask)
    position = 0
    for k in range(m):
        for groupedIndices in groups:
            ls=groupedIndices[k]
            subLen=len(ls)
            if subLen>0:
                for index in ls:
                    argsorted[position] = index
                    position+= 1   
    return argsorted
@njit(fastmath=True,parallel=True)
def arrayCountingArgSortAndUniqueValuesThreaded(array,maxval,mask=np.array([],np.int32),maxThreads=2):    
    m=maxval+1        
    #Allocate output array
    if len(mask)>0:
        arrayLen=len(mask)
    else:
        arrayLen=len(array)
    argsorted=np.empty(arrayLen,np.int32)
    
    #Group indices of same values    
    effectiveSize=int(m*3)        
    if arrayLen<=effectiveSize: 
        nThreads=1
    else:
        nThreads=min(max(arrayLen//effectiveSize,1),maxThreads)            
    groups=[[emptyListOfInts() for k in range(0)]]*nThreads
    chunkSize=arrayLen//nThreads
    #print("nThreads=",nThreads)
    for k in prange(nThreads):
        lBound=chunkSize*k;rBound=chunkSize*(k+1)
        if k==nThreads-1:
            rBound=arrayLen
        groups[k]=BinByUniqueValues(array,lBound,rBound,m,mask)
        
    position = 0
    uniqueValues,uniqueValuesIndices=[],[]
    for k in range(m):
        for groupedIndices in groups:
            if len(groupedIndices[k])>0:
                if not (k in uniqueValues):
                    uniqueValues.append(k)
                    uniqueValuesIndices.append(position)
                for index in groupedIndices[k]:
                    argsorted[position] = index
                    position+= 1   
    return np.array(uniqueValues,np.int32),np.array(uniqueValuesIndices,np.int32),argsorted

def topk_by_partition(input:np.ndarray, k:int, axis:int=None, ascending:bool=False)->tuple:
    """Returns indices and values of TOP-k elements of an array"""
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val