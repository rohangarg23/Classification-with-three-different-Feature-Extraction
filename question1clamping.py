import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
def emojis_to_unicode_points(emoji_string):
    return [ord(char) for char in emoji_string]

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
train_emoticon_X_unicode = [emojis_to_unicode_points(i) for i in train_emoticon_X]
for i in range(0,9):
    for j in range(len(train_emoticon_X_unicode)):
        a=0
        for k in [128529,128539,128547,128623,128700,128720]:
            if train_emoticon_X_unicode[j][i] == k:
                a= 1
                break
        if a ==0:
            train_emoticon_X_unicode[j][i] = 128500
for i in range(9,13):
    for j in range(len(train_emoticon_X_unicode)):
        a=0
        for k in [128529,128547,128623,128636]:
            if train_emoticon_X_unicode[j][i] == k:
                a= 1
                break
        if a ==0:
            train_emoticon_X_unicode[j][i] = 128500
pd.DataFrame(train_emoticon_X_unicode).to_csv("clamp.csv")
X_train,X_test,y_train,y_test=train_test_split(train_emoticon_X_unicode,train_emoticon_Y,test_size=0.2)


model =svm.SVC(kernel="rbf")
model.fit(X_train,y_train)
predictions = model.predict(X_test)
acc =accuracy_score(y_test,predictions)
print("accuracy: ",acc)
