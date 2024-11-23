import numpy as np
import sklearn
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch 

train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']
# train_feat_X_tensor = torch.tensor(train_feat_X)
train_feat_X_flattened  = train_feat_X.reshape(7080,13*768)
X_train,X_test, y_train,y_test = train_test_split(train_feat_X_flattened,train_feat_Y,test_size=0.2)
model = svm.LinearSVC()
model.fit(X_train,y_train)
# print(train_feat_X_flattened.shape)
prediction = model.predict(X_test)
acc=accuracy_score(y_pred=prediction,y_true=y_test)
print(acc)
