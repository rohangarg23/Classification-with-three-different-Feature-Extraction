import numpy as np
import pandas as pd
import sklearn
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import gensim.downloader as api

embedding = api.load('word2vec-google-news-300')
def feature_embedding_maker(arr , embeddings = embedding):
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            arr[i][j] = embedding[arr[i][j]]
        

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
for k in range(len(train_emoticon_X)):
    train_emoticon_X[k] =[a for a in train_emoticon_X[k].split()[0]]
feature_embedding_maker(train_emoticon_X)
train_emoticon_X = np.array(train_emoticon_X)
shape = train_emoticon_X.shape
print(shape)
train_emoticon_X = train_emoticon_X.reshape(shape[0],shape[1]*shape[2])
# print(np.array(train_emoticon_X).shape)
print(train_emoticon_X.shape)
X_train,X_test,y_train,y_test = train_test_split(train_emoticon_X,train_emoticon_Y)
model = svm.SVC()
model.fit(X_train,y_train)
prediction = model.predict(X_test)
acc = accuracy_score(y_test,prediction)
print(acc)

# for i in range(len())

