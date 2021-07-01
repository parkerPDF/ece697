import sys
import sklearn
import numpy as np
from scipy.io import loadmat
import functionList

memNumber = (sys.argv[1]) #unpack from the inputs
ridgeNumber = (sys.argv[2])
whatCv = (sys.argv[3])
fileName = (sys.argv[4])
cvNum = 10

raw = loadmat(fileName)
X = raw['data']  
mask = raw['mask']
l = X.shape[0]

ridgeNormal = np.dot(ridgeNumber,l)

featureMat = functionList.buildFeatureMat(X,memNumber)
xClean = functionList.removeTrash(X,mask,memNumber)
featureMat = functionList.removeTrash(featureMat,mask,memNumber)

crossVal = sklearn.model_selection.KFold(n_splits=cvNum)
folds = [next(crossVal.split(featureMat)) for i in range(cvNum)]
train_in = folds[whatCv][0]
test_in = folds[whatCv][1]

pred = functionList.predRrOutput(featureMat[train_in,:], featureMat[test_in,:], xClean[train_in], ridgeNormal, 0)
error = functionList.findMSE(pred, xClean[test_in])

print(error)