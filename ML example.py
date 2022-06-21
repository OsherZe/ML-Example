## Importing relevant libraries
import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

## Creating a dataframe using pandas
train_df = pd.read_csv('/content/ML project dataframe csv.csv', sep = ",")

print("Number of columns including labels:", len(train_df.columns))
print("Number of rows:", len(train_df))

## Plotting labels histogram to chack balance
x = train_df['target']
ax = sns.countplot(x=x, data=train_df)

## Chacking for missing values in the CSV file
train_df.isnull().any().any()

## Splitting the data to train and test
data_features = train_df.drop(['target'], axis=1) #axis=1 for choosing a column
data_targets = train_df['target']

X_train, X_test, y_train, y_test = train_test_split(data_features, data_targets, test_size= 0.2)

#Converting x,y to numpy arrays
X_train = pd.DataFrame(X_train).to_numpy()
y_train = pd.DataFrame(y_train).to_numpy()
X_test = pd.DataFrame(X_test).to_numpy()
y_test = pd.DataFrame(y_test).to_numpy()

## Creating Dataset class
class BinaryDataset(Dataset):
  def __init__(self, input_array, target_array):
    self.input_array = input_array
    self.target_array = target_array

  def __len__(self):
    return(len(self.input_array))

  def __getitem__(self, idx):
    x = torch.tensor(self.input_array[idx], dtype=torch.float)
    y = torch.tensor(self.target_array[idx], dtype=torch.float)
    return x, y   

## Creating model class
class LinearNetwork(nn.Module):
  def __init__(self):
    #Stacking layers
    super(LinearNetwork, self).__init__()
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(10, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
  #Defining forward pass method
  def forward(self, x):
    return self.linear_relu_stack(x)

## Hyperparameters
BATCH_SIZE = 400
LR = 0.05
prob_threshold = 0.5
num_epochs = 10

## Creating datasets and dataloaders
train_dataset = BinaryDataset(X_train, y_train)
test_dataset = BinaryDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

## Model and backward pass
# Defining the model
model = LinearNetwork().to(device)

## Backward pass and optimization

# Loss function is gettting model(X) and y
loss_fn = nn.BCEWithLogitsLoss()

#Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LR)

## Creating training functions for each epoch
def train_epoch(dataloader, model, loss_fn, optimizer):
  model.train()
  correct = 0
  train_losses = []
  for batch_idx, (X, y) in enumerate(dataloader):
    # Moving features and labels to the required device
    X, y = X.to(device), y.to(device) 
    

   #Computing prediction and loss 
    pred = model(X)  
    loss = loss_fn(pred, y)


    #Zero the gradiants and backward pass
    optimizer.zero_grad()
    loss.backward()

    #Updating weights
    optimizer.step()

    # Creating training loss dictionary
    train_losses.append(loss.item())

    # Creating a correct array which gets the prediction output, wraps it with sigmoid activation function.
    # If the prediction is correct with a probability higher than the threshold then we get a True value, else we get False value.
    # Multiply by 1 to get 0,1 output and compare to the y array.
    # Summing them all up
    correct += (((torch.sigmoid(pred) > prob_threshold) * 1.0) == y).sum()

    #Printing the loss after a chosen number of batches
    if batch_idx % 100 == 0:
      print(f"Current loss: {loss.item()}")

      print(f"Train accuracy: {correct/len(dataloader.dataset)}")
  return train_losses

## Defining test function per epoch
def test_epoch(dataloader, model, loss_fn):
  model.eval()
  test_loss, correct = 0,0
  with torch.no_grad(): #Setting requires_grad to False in every tensor inside the
    for X, y in dataloader:
      #Defining X,y and pred inside the scope
      X, y = X.to(device), y.to(device)
      pred = model(X)

      #Summing the loss of the test into a dictionary
      test_loss +=  loss_fn(pred, y).item()

      #Defining correct the same as in train_epoch
      correct += (((torch.sigmoid(pred) > prob_threshold) * 1) == y).sum()
  print(f"Test Avg loss: {test_loss/len(dataloader)} Test accuracy: {correct/len(dataloader.dataset)}")
  return test_loss/len(dataloader)

## Running the model
EPOCHS = num_epochs
train_losses = []
test_losses = []
for t in range(EPOCHS):
  print(f"Epoch # {t+1} --------------------------------")
  train_losses += train_epoch(train_dataloader, model, loss_fn, optimizer) 
  test_losses.append(test_epoch(test_dataloader, model, loss_fn))

## Plotting test and train losses
# Train Data

x_axis_data = np.arange(len(train_losses)) # X axis data
y_axis_data = train_losses # Y axis data

# Style

graph_title = "Train loss over number of batches" # Main title
x_axis_label = "Num of bachtes" # X axis label
y_axis_label = "Train loss" # Y axis label
plot_style = 'Solarize_Light2' # Choosing plotting style
legend_label = 'Loss' #Legend label

# Plotting the graph

plt.plot(x_axis_data, y_axis_data, label=legend_label) # Applying data
plt.style.use(plot_style) # Enabling plotting style
plt.title(graph_title) # Enabling main title
plt.xlabel(x_axis_label) # Enabling x axis name
plt.ylabel(y_axis_label) # Enabling y axis name
plt.legend(loc='best') # Enabling legend
plt.grid(True) # Enabling grid 
plt.tight_layout() # Compacting the plot 
plt.show() # Graph output

# Test Data

x_axis_data = np.arange(len(test_losses)) # X axis data
y_axis_data = test_losses # Y axis data

# Style

graph_title = "Test loss over number of batches" # Main title
x_axis_label = "Num of epochs" # X axis label
y_axis_label = "Test loss" # Y axis label
plot_style = 'bmh' # Choosing plotting style
legend_label = 'Loss' #Legend label

# Plotting the graph

plt.plot(x_axis_data, y_axis_data, label=legend_label) # Applying data
plt.style.use(plot_style) # Enabling plotting style
plt.title(graph_title) # Enabling main title
plt.xlabel(x_axis_label) # Enabling x axis name
plt.ylabel(y_axis_label) # Enabling y axis name
plt.legend(loc='best') # Enabling legend
plt.grid(True) # Enabling grid 
plt.tight_layout() # Compacting the plot 
plt.show() # Graph output

## Defining sigmoing function output for plotting the overlap 
#train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
train_dataloader = DataLoader(test_dataset, batch_size=1)
def sigmoid_output(train_dataloader, model):
  EPOCHS = num_epochs
  output_0 = []
  output_1 = []
  for c, (X, y) in enumerate(train_dataloader):
  
    # Moving features and labels to the required device
    X, y = X.to(device), y.to(device)
    pred = model(X)

    if y == 1:
      output_1.append(torch.sigmoid(pred))

    else:
      output_0.append(torch.sigmoid(pred))
  
  return output_0, output_1    

## Creating lists of outputs 0,1
output_0, output_1 =  sigmoid_output(train_dataloader, model)
output_0 = [x.item() for x in output_0]
output_1 = [x.item() for x in output_1]

## Plotting the distribution of outputs
plt.title("Distribution of outputs")
plt.xlabel("Output")
plt.ylabel("Num of examples")
plt.hist(output_0, bins=1000, histtype='step', label="Label 0 distribution", range=(0,1))
plt.hist(output_1, bins=1000, histtype='step', color='red', label="Label 1 distribution", range=(0,1))
plt.legend(loc='best')
plt.show()