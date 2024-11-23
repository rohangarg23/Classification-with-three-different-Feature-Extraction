import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example data: 2D points and their binary labels
# X is a 2D array of points
# y is the label array (0 or 1)
def emojis_to_unicode_points(emoji_string):
    return [ord(char) for char in emoji_string]

train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
train_emoticon_X_unicode = [emojis_to_unicode_points(i) for i in train_emoticon_X]
for v in range(13):
    for w in range(v,13):


        train_emoticon_X_unicode_first_2vectors = [[i[v],i[w]] for i in train_emoticon_X_unicode ]
        # print(train_emoticon_X_unicode_first_2vectors)

        # Example data: 2D points and their binary labels
        X = np.array(train_emoticon_X_unicode_first_2vectors)
        y = np.array(train_emoticon_Y)

        # Separate the points based on labels
        class_0 = X[y == 0]
        class_1 = X[y == 1]

        # Create subplots: 1 row, 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Plot class 0 points on the first subplot
        ax1.scatter(class_0[:, 0], class_0[:, 1], color='red', label='Class 0')
        ax1.set_title('Class 0 Data')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')

        # Plot class 1 points on the second subplot
        ax2.scatter(class_1[:, 0], class_1[:, 1], color='blue', label='Class 1')
        ax2.set_title('Class 1 Data')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')

        # Adjust layout
        plt.tight_layout()

        # Show the plots
        # plt.show()

        #Save the plot
        plt.savefig("Figure"+str(v+1)+"and"+str(w+1), dpi=300)
