import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LinearModelWithBias(nn.Module):
    def __init__(self):
        super(LinearModelWithBias, self).__init__()
        # Initialize your model parameters here
        self.hidden1 = nn.Linear(1, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100, bias=True)
        self.hidden3 = nn.Linear(100, 100, bias=True)
        self.output = nn.Linear(100, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Runs the model for a batch of examples
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.relu(self.hidden3(x))
        predicted_y = self.output(x)
        return predicted_y

    def get_loss(self, x, y):
        # Computes the loss for a batch of examples
        loss_function = nn.MSELoss()
        loss = loss_function(self.forward(x), y)
        return loss

    def train(self, dataset, batch_size=50, num_epochs=1000, learning_rate=0.001):
        # Trains the model
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for batch in dataloader:
                x_batch, y_batch = batch['x'], batch['label']
                optimizer.zero_grad()
                loss = self.get_loss(x_batch, y_batch)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Example usage:
if __name__ == "__main__":
    # Generate training data
    x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
    y_train = np.sin(x_train)
    dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32).view(-1, 1),
                                             torch.tensor(y_train, dtype=torch.float32).view(-1, 1))

    # Initialize and train the model
    model = LinearModelWithBias()
    model.train(dataset, batch_size=50)  # Specify the batch size here

    # Evaluate the trained model
    x_test = np.linspace(-2 * np.pi, 2 * np.pi, 500)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)
    with torch.no_grad():
        y_pred = model(x_test_tensor).numpy()

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_train, y_train, label="True sin(x)")
    plt.plot(x_test, y_pred, label="Approximated sin(x)", linestyle="--", color="r")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.title("Approximating sin(x) using a Linear Neural Network with Bias")
    plt.legend()
    plt.grid(True)
    plt.show()
