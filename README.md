# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
This experiment focuses on building a Neural Network regression model to predict continuous values from input data. It involves preprocessing the dataset through normalization and feature scaling, followed by designing a neural network with appropriate layers and activation functions.

The model is trained using backpropagation and gradient descent, and its performance is evaluated using metrics such as MSE, MAE, and R² score. The study also explores how hyperparameters like learning rate and network depth influence accuracy, while applying validation and regularization techniques to ensure better generalization.

Overall, the experiment provides practical insight into applying deep learning for real-world regression tasks.


## Neural Network Model
<img width="1100" height="699" alt="image" src="https://github.com/user-attachments/assets/5e8bddd0-cc0a-4aa2-911b-71a4351750b7" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: VISHAL P

### Register Number: 212224230306

```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset1 = pd.read_csv('/content/drive/MyDrive/Deep Learning exp/DL EXP-1spreadsheet - Sheet1 (2).csv')
X = dataset1[['INPUT']].values
y = dataset1[['OUTPUT']].values
dataset1.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here
     for epoch in range (epochs):
        optimizer. zero_grad()
        loss=criterion(ai_brain(X_train), y_train)
        loss. backward()
        optimizer.step()
        lig .history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)
with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')
loss_df = pd.DataFrame(lig.history)
import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()
X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')

```

### Dataset Information
<img width="216" height="172" alt="image" src="https://github.com/user-attachments/assets/b078d545-5e6a-41f0-9b83-cb6bf4b87235" />



### OUTPUT
<img width="251" height="153" alt="image" src="https://github.com/user-attachments/assets/16f81fd3-2dbd-49cd-b381-7145042f5e4f" />


### Training Loss Vs Iteration Plot
<img width="487" height="383" alt="image" src="https://github.com/user-attachments/assets/197e7d42-dee5-46b3-ab7d-4aa87630a0dc" />



### New Sample Data Prediction
<img width="228" height="31" alt="image" src="https://github.com/user-attachments/assets/928e23f9-2338-4887-a8e2-bf7815cb571d" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
