# EmoMA-Net
## Our Paper: 

## EmoMA-Net Model Training and Evaluation
### Overview
This project aims to train and evaluate a deep learning model for emotion recognition based on the WESAD dataset [1]. The model architecture combines Convolutional Neural Networks (CNN) with Convolutional Block Attention Module (CBAM) and Long Short-Term Memory (LSTM) networks to process time-series data effectively [2, 3, 4].
### Dataset
The dataset used for training and evaluation is stored in a CSV file named `merged.csv`. It contains time-series data along with labels representing different emotions. Before training, the dataset is preprocessed to ensure compatibility with the model.
### Data Preparation
1. Load and Preprocess Data:
- Load the dataset from the CSV file.
- Change the labels to binary values (0 or 1).
- Select relevant features for training.
- Scale the features using `StandardScaler`.
- Perform Recursive Feature Elimination (RFE) to select the top features.
2. Split Data:
- Split the data into training and testing sets.
- Use KFold for cross-validation.
### Model Definition
The model consists of a CNN for feature extraction followed by an LSTM layer for sequence modeling. The model architecture is defined as follows:
- Input Layer: Accepts input data of shape (batch_size, channels, height, width).
- CNN Layers: Multiple convolutional layers to extract features.
- CBAM Layers: Spatial Attention and Channel Attention.
- LSTM Layer: Processes the extracted features over time.
- Output Layer: Produces the final classification output.
### Training Process
1. Initialize Model and Optimizer:
- Define the model and optimizer (Adam).
- Set the loss function (CrossEntropyLoss).
2. Training Loop:
- Train the model over multiple epochs.
- Calculate the loss and accuracy during each epoch.
- Track the best model based on the validation accuracy.
### Evaluation
1. Testing Loop:
- Evaluate the model on the test set.
- Calculate the final test accuracy.
- Generate a confusion matrix to assess performance.
### Running the Code
To run the code, follow these steps:
1. Install Dependencies:
Ensure you have Python installed.
Install required libraries using pip:
```python
     pip install torch pandas scikit-learn numpy
```
2. Prepare the Dataset:
- Place the merged.csv file in the specified directory.
- Modify the path to the CSV file in the code if necessary.
3. Execute the Script:
Run the script using Python:
```python
     python EmoMA-Net.py
```
### Results
The script prints out the training progress, including the loss and accuracy at each epoch. After training, it displays the maximum test accuracy and f1-score achieved across all folds of cross-validation.
### License
This project is licensed under the MIT License - see the LICENSE file for details.
### Note
Make sure to adjust the file paths and other settings according to your specific environment and requirements.

### REFERENCE
[1] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. 2018. Introducing WESAD, a multimodal dataset for wearable stress and affect detection. Proceedings of the 20th ACM International Conference on Multimodal Interaction (October 2018). DOI:http://dx.doi.org/10.1145/3242969.3242985. 
[2] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So Kweon. 2018a. CBAM: Convolutional Block Attention Module. (July 2018). Retrieved July 24, 2024 from https://arxiv.org/abs/1807.06521
[3] Liu, Y., Kang, J., Li, Y., & Ji, B. (2021). A network intrusion detection method based on CNN and CBAM. IEEE INFOCOM 2021 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS). https://doi.org/10.1109/infocomwkshps51825.2021.9484553 
[4] Maximilian Beck et al. 2024. XLSTM: Extended Long Short-term memory. (May 2024). Retrieved July 24, 2024 from https://arxiv.org/abs/2405.04517
