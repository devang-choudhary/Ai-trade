import torch
import torch.nn as nn


# Define the LSTM model class
class Stock_predictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,):
        super(Stock_predictor, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 25)
        self.fc2 = nn.Linear(25, 1)
        
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out

# Instantiate the model
input_size = 1  # Assuming your input size is 1
hidden_size1 = 128
hidden_size2 = 64
model = Stock_predictor(input_size, hidden_size1, hidden_size2)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# Training loop
num_epochs = 1
batch_size = 1
for epoch in range(num_epochs):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(x_train)}], Loss: {loss.item():.4f}')