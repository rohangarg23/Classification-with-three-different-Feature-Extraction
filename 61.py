import numpy as np
import pandas as pd
import sklearn
import sklearn.svm as svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model 
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Input,Embedding,Dense,Conv1D
from tensorflow.keras.saving import load_model
import matplotlib.pyplot as plt
from scipy import stats

# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        NotImplemented
    
    def predict(self, X):
        NotImplemented
    
class TextSeqModel(MLModel):
    def __init__(self) -> None:
        #loading the datasets
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        train_seq_X = train_seq_df['input_str'].tolist()
        train_seq_Y = train_seq_df['label'].tolist()
        val_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
        val_seq_X = val_seq_df['input_str'].tolist()
        val_seq_Y = val_seq_df['label'].tolist()
        #converting into numpy arrays for easier preprocessing
        x=np.array(train_seq_X)
        x_val=np.array(val_seq_X)
        #converting labels of training and validation data into numpy array
        y=np.array(train_seq_Y)
        y_val=np.array(val_seq_Y)
        # Convert each element to char, then to corresponding int
        t =np.array([list(s) for s in x])  
        t_val=np.array([list(s) for s in x_val])
        # changing the data type of each element to int
        t=t.astype(int)
        t_val=t_val.astype(int)
        # a sample of what a training example looks like after above preprocessing
        print(f"A sample training example of 50 timesteps: {t[0]}")

        #Creating a Keras sequential model
        self.model=Sequential()
        self.model.add(Input(shape=(50,)))
        self.model.add(Embedding(input_dim=10,output_dim=16))  #embedding layer capture semantic information
        self.model.add(Conv1D(filters=64, kernel_size=3, activation='relu')) #1D convolution layer to capture local patterns in the sequence
        self.model.add(LSTM(16))
        self.model.add(Dense(16,activation='relu'))
        self.model.add(Dense(16,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid')) #output layer

        self.model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

        #model summary showing number of trainable parameters
        self.model.summary()

        #run the below code to fit the model
        # history=self.model.fit(t,y,epochs=25,validation_data=(t_val,y_val))

        #loading a model already trained to save training time
        self.model=load_model('final.keras')

        #run the following code to plot accuracy vs epoch plot i.e accuracy after each epoch
        # plt.plot(history.history['accuracy'],label='Training accuracy')
        # plt.plot(history.history['val_accuracy'],label='Validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy(%)')
        # plt.legend()
        # plt.show()

        y_val_pre=np.squeeze(self.model.predict(t_val)>0.5).astype(int)
        print(f"Accuracy on 3rd datset: {accuracy_score(y_val_pre,y_val)}")
    def predict(self, X):
        x_pre=np.array(X)
        t_pre=np.array([list(s) for s in x_pre])
        t_pre=t_pre.astype(int)
        y_pre=np.squeeze(self.model.predict(t_pre)>0.5).astype(int)
        return y_pre
        
    
    
class EmoticonModel(MLModel):
    def __init__(self) -> None:
       # Step 1: Load Emoji Dataset (13-emoji sequences as input, labels as output)
        train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
        train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
        train_emoticon_Y = train_emoticon_df['label'].tolist()
        test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
        valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
        valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
        valid_emoticon_Y = valid_emoticon_df['label'].tolist()
        c = train_emoticon_X+test_emoticon_X

        # Example emoji vocabulary (you will use your dataset)
        emoji_vocab = list(set([emoji for seq in c for emoji in seq]))
        emoji_to_idx = {emoji: idx for idx, emoji in enumerate(emoji_vocab)}
        vocab_size = len(emoji_vocab)
        embedding_dim = 7 # Size of each emoji vector
        # emoji_to_idx.update([['\U0001f6da',214],['\U0001f675',215],['\U0001f66D',216],['\U0001f6E0',217]])
        self.emoji_to_idx = emoji_to_idx
        # print(emoji_to_idx['\U0001F61B'])
        print(self.emoji_to_idx)
        self.vocab_size = vocab_size
        self.emembedding_dim = embedding_dim

        # Function to convert emoji sequences into indices
        def emoji_to_index_sequence(emoji_seq, emoji_to_idx):
            return [emoji_to_idx[emoji] for emoji in emoji_seq]

        # Convert all emoji sequences into their corresponding indices
        emoji_sequences = [emoji_to_index_sequence(seq, emoji_to_idx) for seq in train_emoticon_X]
        emoji_sequences_valid = [emoji_to_index_sequence(seq, emoji_to_idx) for seq in valid_emoticon_X]


        # Convert data to PyTorch tensors
        emoji_sequences = torch.tensor(emoji_sequences, dtype=torch.long)
        emoji_sequences_valid = torch.tensor(emoji_sequences_valid, dtype=torch.long)
        train_labels = torch.tensor(train_emoticon_Y, dtype=torch.long)
        valid_labels = torch.tensor(valid_emoticon_Y, dtype=torch.long)

        # Step 2: Define the PyTorch Model with an Embedding Layer and a Linear Layer
        class EmojiEmbeddingModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, num_classes):
                super(EmojiEmbeddingModel, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.fc = nn.Linear(embedding_dim * 13, num_classes)  # Fully connected layer for classification
            
            def forward(self, x):
                # Embedding output shape will be (batch_size, 13, embedding_dim)
                embeddings = self.embedding(x)
                # Flatten to (batch_size, 13 * embedding_dim)
                embeddings = embeddings.view(embeddings.size(0), -1)
                # Pass through the linear layer to get logits for each class
                logits = self.fc(embeddings)
                return logits

        # Step 3: Initialize and train the model
        num_classes = len(set(train_emoticon_Y))  # Number of classes based on your dataset
        model = EmojiEmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim, num_classes=num_classes)

        # Set loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        # Sample training loop (you can adjust the number of epochs and batch size)
        num_epochs = 30
        batch_size = 32

        for epoch in range(num_epochs):
            model.train()
            permutation = torch.randperm(emoji_sequences.size(0))
            
            for i in range(0, emoji_sequences.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = emoji_sequences[indices], train_labels[indices]

                # Forward pass
                optimizer.zero_grad()
                logits = model(batch_x)  # Now it returns logits for each class
                loss = loss_fn(logits, batch_y)  # Compute the loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # Step 4: Extract embeddings after training
        def get_embeddings(model, emoji_sequences):
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                embeddings = model.embedding(emoji_sequences)  # Get embeddings
            flattened_embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten to (num_samples, 13 * embedding_dim)
            return flattened_embeddings

        flattened_embeddings = get_embeddings(model, emoji_sequences)  # Shape will be (num_samples, 208)
        flattened_embeddings_valid = get_embeddings(model,emoji_sequences_valid)


        # Step 5: Split data for SVM training
        X_train, X_test, y_train, y_test = train_test_split(flattened_embeddings.numpy(), train_labels.numpy(), test_size=0.01, random_state=42)
        X_valid = flattened_embeddings_valid.numpy()
        y_valid = valid_labels.numpy()

        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_test_scaled = scaler.transform(X_test)


        svm_model = svm.LinearSVC()
        svm_model.fit(X_train, y_train)
        self.svm_model = svm_model 

        pred = svm_model.predict(X_valid)
        accuracy = accuracy_score(y_valid, pred)
        print(f"Accuracy on 1st dataset: {accuracy}")
        self.model = model
    def get_embeddings(self, emoji_sequences):
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                embeddings = self.model.embedding(emoji_sequences)  # Get embeddings
            flattened_embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten to (num_samples, 13 * embedding_dim)
            return flattened_embeddings

    def emoji_to_index_sequence(self,emoji_seq):
            return [self.emoji_to_idx[emoji] for emoji in emoji_seq]
    def predict(self, X): # random predictions
        emoji_sequences = [self.emoji_to_index_sequence(seq) for seq in X]
        emoji_sequences = torch.tensor(emoji_sequences, dtype=torch.long)
        X = self.get_embeddings(emoji_sequences)
        
        return self.svm_model.predict(X)
    
class FeatureModel(MLModel):
    def __init__(self) -> None:
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat['features']
        train_feat_Y = train_feat['label']
        valid_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
        valid_feat_X = valid_feat['features']
        valid_feat_Y = valid_feat['label']
        # train_feat_X_tensor = torch.tensor(train_feat_X)
        train_feat_X_flattened  = train_feat_X.reshape(7080,13*768)
        valid_feat_X_flattened = valid_feat_X.reshape(len(valid_feat_X), 13*768)
        X_train = train_feat_X_flattened
        y_train = train_feat_Y
        # X_train,X_test, y_train,y_test = train_test_split(train_feat_X_flattened,train_feat_Y,test_size=0.8)
        X_test  = valid_feat_X_flattened
        y_test = valid_feat_Y
        #To run linear SVM model uncomment the below line and comment the next line
        # model = svm.SVC(kernel='linear')
        model = linear_model.LogisticRegression()
        model.fit(X_train,y_train)
        # print(train_feat_X_flattened.shape)
        prediction = model.predict(X_test)
        acc=accuracy_score(y_pred=prediction,y_true=y_test)
        print(f"Accuracy on 2nd dataset: {acc}")
        self.model = model
    def predict(self, X):# random predictions
        X =X.reshape(len(X), 13*768)
        return self.model.predict(X)
class CombinedModel(MLModel):
    def __init__(self, feature_model, emoticon_model, text_model) -> None:

        self.feature_model = feature_model
        self.emoticon_model = emoticon_model
        self.text_model = text_model
        #load 3 datasets
        valid_emoticon_X = pd.read_csv("datasets/valid/valid_emoticon.csv")['input_emoticon'].tolist()
        valid_seq_X = pd.read_csv("datasets/valid/valid_text_seq.csv")['input_str'].tolist()
        valid_feat_X = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)['features']
        # load validation labels
        y_val = np.array(pd.read_csv("datasets/valid/valid_emoticon.csv")['label'])
         # Get individual model predictions
        pre_feat = self.feature_model.predict(valid_feat_X)
        pre_emoticons = self.emoticon_model.predict(valid_emoticon_X)
        pre_text = self.text_model.predict(valid_seq_X)
        # Stack predictions for hard voting
        combined_preds = np.vstack([pre_text, pre_emoticons, pre_feat])
        # Use mode (hard voting) to get the majority class
        final_predictions = stats.mode(combined_preds, axis=0)[0].squeeze()
        print(f"Validation accuracy after combining 3 datasets: {accuracy_score(y_val,final_predictions)}")
        
    def predict(self, X1, X2, X3): # random predictions
        # Get individual model predictions
        pred_feat = self.feature_model.predict(X1)
        pred_emoticons = self.emoticon_model.predict(X2)
        pred_text = self.text_model.predict(X3)
        # Stack predictions for hard voting
        combined_preds = np.vstack([pred_feat, pred_emoticons, pred_text])
        
        # Use mode (hard voting) to get the majority class
        final_predictions = stats.mode(combined_preds, axis=0)[0].squeeze()
        return final_predictions
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    emoticon_model  = EmoticonModel()
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    best_model = CombinedModel(feature_model, emoticon_model, text_model)
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
