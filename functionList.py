### use 'import functionList' to get this into your ipynb file
#### make sure the files are in the same directory

import numpy as np

#l is length of the matrix and it's optional
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

########### With these two functions you should be able to solve RR with:
"""
memMat = functionList.buildFeatureMat(X,numMemoryPoints) # build reature vector
ols = Ridge(alpha=0) ###
ols.fit(memMat,X) ### replace these three line with however you want to solve RR
pred = ols.predict(memMat)###
functionList.findMSE(pred,X) # get MSE
"""

