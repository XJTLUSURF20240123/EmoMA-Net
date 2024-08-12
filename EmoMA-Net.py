import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# Custom dataset class for loading WESADDataset
class WESADDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe.drop('subject', axis=1)
        self.labels = self.dataframe['label'].values
        self.dataframe.drop('label', axis=1, inplace=True)

    def __getitem__(self, idx):
        x = self.dataframe.iloc[idx].values
        x = x.reshape(1, -1)  # Adjust the input shape to CNN
        y = self.labels[idx]
        return torch.Tensor(x), y

    def __len__(self):
        return len(self.dataframe)

# Define the list of features
feats = ['BVP_mean', 'BVP_std', 'BVP_min', 'BVP_max',
         'EDA_phasic_mean', 'EDA_phasic_std', 'EDA_phasic_min', 'EDA_phasic_max', 'EDA_smna_mean',
         'EDA_smna_std', 'EDA_smna_min', 'EDA_smna_max', 'EDA_tonic_mean',
         'EDA_tonic_std', 'EDA_tonic_min', 'EDA_tonic_max', 'Resp_mean',
         'Resp_std', 'Resp_min', 'Resp_max', 'TEMP_mean', 'TEMP_std', 'TEMP_min',
         'TEMP_max', 'TEMP_slope', 'BVP_peak_freq', 'age', 'height',
         'weight', 'subject', 'label']

# Function to get data loaders for training and testing
def get_data_loaders(df, train_subjects, test_subjects, train_batch_size=25, test_batch_size=5):

    # Split the training and test sets based on randomly selected people
    train_df = df[df['subject'].isin(train_subjects)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subjects)].reset_index(drop=True)

    # Create data loaders for the training and test sets
    train_dset = WESADDataset(train_df)
    test_dset = WESADDataset(test_df)

    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dset, batch_size=test_batch_size)

    return train_loader, test_loader

# Function to calculate the output size of the convolutional layer
def calculate_conv_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1

# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(lstm_hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # Computing attention weights
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        # Weighted average LSTM output using attention weights
        attn_output = torch.bmm(attn_weights.transpose(1, 2), lstm_output).squeeze(1)
        return attn_output

# Define the Convolutional Block Attention Module (CBAM)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_wise = self.channel_gate(x) * x
        spatial_wise = self.spatial_gate(x) * x
        return channel_wise + spatial_wise

# Define the CNN-LSTM model with attention and CBAM
class CNNLSTMModel(nn.Module):
    def __init__(self, lstm_hidden_dim=50, num_lstm_layers=1):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.cbam1 = CBAM(in_channels=16)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.cbam2 = CBAM(in_channels=32)

        # Calculate the dimension of the output of the convolutional layer
        conv_output_dim = calculate_conv_output_dim(10, 3, 1, 1) // 4

        # Compute the convolutional layer outputs Compute the dimension of the output size of the convolutional layer
        self.conv_output_size = 32 * conv_output_dim

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=self.conv_output_size, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)

        # Defining the attention layer
        self.attention = Attention(lstm_hidden_dim)

        # Define the fully connected layer
        self.fc1 = nn.Linear(lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 3)  # Make sure the output dimension is 3

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Part of CNN
        x = F.relu(self.conv1(x))
        x = self.cbam1(x)  # Adding CBAM
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.cbam2(x)  # Adding CBAM
        x = self.pool(x)

        # The shape is adjusted to fit the input of the LSTM layer
        x = x.view(x.size(0), 1, -1)  # (batch_size, sequence_length=1, input_size=conv_output_size)

        # LSTM part
        lstm_out, _ = self.lstm(x)

        # Attention mechanism
        attn_output = self.attention(lstm_out)

        # Fully connected layer
        x = F.relu(self.fc1(attn_output))
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

# Function to train the model
def train(model, optimizer, train_loader, validation_loader):
    # Initialize a dictionary that records the loss and accuracy during training and validation
    history = {'train_loss': {}, 'train_acc': {}, 'valid_loss': {}, 'valid_acc': {}}
    # Training the model
    for epoch in range(num_epochs):
        total = 0
        correct = 0
        trainlosses = []

        for batch_index, (images, labels) in enumerate(train_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Loss
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trainlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

        history['train_loss'][epoch] = np.mean(trainlosses)
        history['train_acc'][epoch] = correct / total

        if epoch % 10 == 0:
            with torch.no_grad():

                losses = []
                total = 0
                correct = 0

                for images, labels in validation_loader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images.float())
                    loss = criterion(outputs, labels)

                    # Compute accuracy
                    _, argmax = torch.max(outputs, 1)
                    correct += (labels == argmax).sum().item()  # .mean()
                    total += len(labels)

                    losses.append(loss.item())

                history['valid_acc'][epoch] = np.round(correct / total, 3)
                history['valid_loss'][epoch] = np.mean(losses)

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(losses):.4}, Acc: {correct / total:.2}')

    return history

# Define the function to test the model
def test(model, validation_loader):
    print('Evaluating model...')
    # Test
    model.eval()

    total = 0
    correct = 0
    testlosses = []
    correct_labels = []
    predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(validation_loader):
            # Send to GPU (device)
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images.float())

            # Compute actual probabilities
            probabilities = torch.exp(outputs)

            # Loss
            loss = criterion(outputs, labels)

            testlosses.append(loss.item())

            # Compute accuracy
            _, argmax = torch.max(outputs, 1)
            correct += (labels == argmax).sum().item()  # .mean()
            total += len(labels)

            correct_labels.extend(labels)
            predictions.extend(argmax.cpu())

    test_loss = np.mean(testlosses)
    accuracy = np.round(correct / total, 2)
    print(f'Loss: {test_loss:.4}, Acc: {accuracy:.2}')

    # Convert to numpy arrays for F1 score calculation
    y_true = np.array([label.item() for label in correct_labels])
    y_pred = np.array([label.item() for label in predictions])

    f1 = f1_score(y_true, y_pred, average='binary')  # For binary classification
    print(f'F1 Score: {f1:.2}')

    cm = confusion_matrix(y_true, y_pred)
    return cm, test_loss, accuracy, f1

# Load the dataset
df = pd.read_csv('data\merged.csv', index_col=0)
# Get all subject IDs
subject_id_list = df['subject'].unique()

# Define a function to change labels, converting non-0 or non-1 labels to 1
def change_label(label):
    if label == 0 or label == 1:
        return 0
    else:
        return 1

# Apply the label-changing function to the label column of the dataset
df['label'] = df['label'].apply(change_label)

# Select feature columns
X = df[feats[:-2]]  # Exclude 'subject' and 'label' columns
y = df['label']

# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler object using the training set data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform the test set data using the same scaler object
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Create an RFE object, specifying the Random Forest model and the target number of features to select
rfe = RFE(estimator=rf, n_features_to_select=10)

# Fit the data and obtain the selected features
rfe.fit(X_train_scaled, y_train)

# Get the indices of the selected features
selected_features_index = rfe.support_

# Use the selected feature indices to obtain the selected features
selected_features = X.columns[selected_features_index]

print("Top features selected by RFE:")
print(selected_features)

# Rebuild the dataset using the selected features
df = df[selected_features.tolist() + ['label', 'subject']]

# Set the batch sizes for training and testing
train_batch_size = 25
test_batch_size = 5

# Set the device, preferring GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the number of training epochs
num_epochs = 100

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Initialize lists to store results
histories = []
confusion_matrices = []
test_losses = []
test_accs = []

# Set the number of folds for cross-validation
num_folds = 2
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_count = 0  # Initialize the fold counter

# Initialize the maximum accuracy and the best model
max_acc = 0.0
f1_scores = []
best_model = None
import copy

for train_index, test_index in kf.split(df):
    fold_count += 1
    print(f'Final training and testing - Fold {fold_count}:')  # Add this line to print the current fold number
    train_df, test_df = df.iloc[train_index], df.iloc[test_index]

    train_loader, test_loader = get_data_loaders(df, train_df['subject'].unique(), test_df['subject'].unique())

    model = CNNLSTMModel(lstm_hidden_dim=100, num_lstm_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = train(model, optimizer, train_loader, test_loader)
    histories.append(history)

    cm, test_loss, accuracy, f1 = test(model, test_loader)
    test_losses.append(test_loss)
    test_accs.append(accuracy)
    f1_scores.append(f1)

    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total

    # Check if the current model is better than the previous ones
    if test_acc > max_acc:
        max_acc = test_acc
        best_model = copy.deepcopy(model)
        print(f'New best model found with accuracy: {max_acc:.4f}')

# Print the maximum test accuracy and f1-score
print(f'Maximum test accuracy over {num_folds} folds: {max_acc:.4f}')
max_f1 = max(f1_scores)
print(f'Maximum F1-score over {num_folds} folds: {max_f1:.4f}')
