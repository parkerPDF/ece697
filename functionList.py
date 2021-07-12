### use 'import functionList' to get this into your ipynb file
#### make sure the files are in the same directory

import numpy as np
import torch
import sklearn
import sklearn.model_selection
import sklearn.linear_model


#l is length of the matrix and it's optional
#finds the mean square error between two sets
def findMSE(prediction,actual,l=0):
    if l == 0:
        l = len(actual)
    squareError = np.linalg.norm(prediction-actual)
    mse = squareError/l
    return mse

#build the feauture vector make sure X is such that each column is one channel and each row is a different time point
#numMemory points needs to be a positive integer
def buildFeatureMat(X,numMemoryPoints):
    hold = X

    hold = np.roll(hold,1,axis=0)
    hold[0] = 0
    memMat = hold
    i = 1
    if i >= numMemoryPoints: #I couldn't figure out a great way to handle this case.
        return memMat
    for i in range(1,numMemoryPoints):
        hold = np.roll(hold,1,axis=0)
        hold[0] = 0
        memMat = np.hstack((memMat,hold))
    return memMat

## this is only for solving the closed form when the parameters are the columns and rows are the number of data points
## output is going to be a tensor. np -> tensor use torch.from_numpy
## tensor -> np array use tensor method: tensorMatrix.numpy()
###### if l = 0 we can happen accross non-invertale matricies
def solveRrClosedForm(X,Y,l=.1):
    Y = torch.from_numpy(Y)
    X = torch.from_numpy(X)
    X = X.to('cuda')#this is a propert to set in torch to make calc run on GPU
    Y = Y.to('cuda')
    Xt = torch.transpose(X,0,1) #precalculate all the weird torch stuff to avoid errors

    eyeSize = X.shape[1]
    out = torch.matmul(Xt,X) #matmul for 2d matricies are the same as inner product mat multiplication
    out = out + (torch.eye(eyeSize)*l).to('cuda')
    out = out.inverse()
    out = torch.matmul(out,Xt)
    out = torch.matmul(out,Y)
    return out

def predClosedForm(A,w): #handels the cuda properties to do the wX multiplication
    A = torch.from_numpy(A)
    A = A.to('cuda')
    out = torch.matmul(A,w)
    out = out.cpu().detach().numpy()
    return out

#with an X matrix and a mask vector that identifies which data points are usable and which ones are not.
def removeTrash(x, mask, memNumber):
    maskPass = 1
    indexTracking = 0
    x_hold = np.zeros(x.shape) 
    for i in range(x.shape[0]):
        maskPass = 1
        if mask[i] == 0: #if this point is bad don't use it
            maskPass = 0
        else:
            for p in range(1,memNumber+1): #check all of the data points that we are going to use for the feature
                if i-p < 0: #if there are no past time points we can use this point
                    maskPass = 1
                else: #if any of the points in range are bad don't use it.
                    if mask[i-p] == 0:
                        maskPass = 0
        if maskPass == 1:
            x_hold[indexTracking] = x[i,:]
            indexTracking = indexTracking + 1 #keep track of how long the file actually is.

    out = x_hold[0:indexTracking,:]
    return out

#easy way to get the predicted output, can add extra cases as we need.
def predRrOutput(Atrain, Atest, Ytrain, regularizer, useClosedForm=0):
    if useClosedForm == 0:
        RR = sklearn.linear_model.Ridge(alpha=regularizer)
        RR.fit(Atrain,Ytrain)
        pred = RR.predict(Atest)
    else:
        w = solveRrClosedForm(Atrain,Ytrain,regularizer)
        pred = predClosedForm(Atest,w)
    return pred
