import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Step 1: Load Emoji Dataset (13-emoji sequences as input, labels as output)
train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()

valid_emoticon_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon'].tolist()
valid_emoticon_Y = valid_emoticon_df['label'].tolist()


# Example emoji vocabulary (you will use your dataset)
emoji_vocab = list(set([emoji for seq in train_emoticon_X for emoji in seq]))
emoji_to_idx = {emoji: idx for idx, emoji in enumerate(emoji_vocab)}
vocab_size = len(emoji_vocab)
embedding_dim = 7 # Size of each emoji vector

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
num_classes = 2
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
X_train, X_test, y_train, y_test = train_test_split(flattened_embeddings.numpy(), train_labels.numpy(), test_size=0.2, random_state=42)
X_valid = flattened_embeddings_valid.numpy()
y_valid = valid_labels.numpy()

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


svm_model = svm.LinearSVC()
svm_model.fit(X_train, y_train)


pred = svm_model.predict(X_valid)
accuracy = accuracy_score(y_valid, pred)
print(f"Accuracy: {accuracy}")
