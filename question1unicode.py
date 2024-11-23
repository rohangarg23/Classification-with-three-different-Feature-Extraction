import pandas as pd
import matplotlib

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
def emojis_to_unicode_points(emoji_string):
    return [ord(char) for char in emoji_string]

def merge_sort(A, a = 0, b = None,index = 0): # Sort sub-array A[a:b]
    if b is None: # O(1) initialize
        b = len(A) # O(1)
    if 1 < b - a: # O(1) size k = b - a
        c = (a + b + 1) // 2 # O(1) compute center
        merge_sort(A, a, c,index) # T(k/2) recursively sort left
        merge_sort(A, c, b,index) # T(k/2) recursively sort right
        L, R = A[a:c], A[c:b] # O(k) copy
        i, j = 0, 0 # O(1) initialize pointers
        while a < b: # O(n)
            if (j >= len(R)) or (i < len(L) and L[i] <= R[j]): # O(1) check side
                A[a] = L[i] # O(1) merge from left
                i = i + 1 # O(1) decrement left pointer
            else:
                A[a] = R[j] # O(1) merge from right
                j = j + 1 # O(1) decrement right pointer
            a = a + 1 # O(1) decrement merge pointer

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label']
train_emoticon_X_class0 = train_emoticon_X[train_emoticon_Y==0].to_list()
train_emoticon_X_class1 = train_emoticon_X[train_emoticon_Y==1].to_list()
train_emoticon_Y = train_emoticon_Y.to_list()
train_emoticon_X = train_emoticon_X.to_list()
print(len(train_emoticon_X))
print(len(train_emoticon_X_class0))
train_emoticon_X_unicode = [emojis_to_unicode_points(i) for i in train_emoticon_X]
train_emoticon_X_unicode_class0 = [emojis_to_unicode_points(i) for i in train_emoticon_X_class0]
train_emoticon_X_unicode_class1 = [emojis_to_unicode_points(i) for i in train_emoticon_X_class1]
# pd.DataFrame(train_emoticon_X_unicode_class0).to_csv("modi3.csv",index=False)
# pd.DataFrame(train_emoticon_X_unicode).to_csv("modi4.csv",index=False)
for h in range(12,-1,-1):
    merge_sort(train_emoticon_X_unicode_class0,index=h)
    merge_sort(train_emoticon_X_unicode_class1,index=h)
# print(train_emoticon_X_unicode)
pd.DataFrame(train_emoticon_X_unicode_class0).to_csv("modi0.csv",index=False)
pd.DataFrame(train_emoticon_X_unicode_class1).to_csv("modi1.csv",index=False)
# for v in range(13):
#     for w in range(v+1,13):
#         for x in range (w+1,13):

#             train_emoticon_X_unicode_first_2vectors = [[i[v],i[w],i[x]] for i in train_emoticon_X_unicode ]
#             X_train = train_emoticon_X_unicode_first_2vectors
#             y_train = train_emoticon_Y
#             model =svm.SVC()
#             model.fit(X_train,y_train)
#             predictions = model.predict(X_train)
#             acc =accuracy_score(y_train,predictions)
#             print("accuracy: "+"feature"+str(v+1)+"and"+str(w+1)+"and"+str(x+1),acc)