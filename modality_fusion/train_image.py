import os
import random
import torch
import numpy as np
import pandas as pd
import pickle5 as pickle
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 100, kernel_size=3, activation='relu')
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(100, 50, kernel_size=3, activation='relu')
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(50 * 17 * 17, 3)  # Adjust the input features according to the output of the last MaxPooling layer

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.dropout1(x)
        x = self.pool2(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def main():
    # ... (Data loading and preprocessing code remains mostly unchanged)

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = CNNModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(50):
            for i, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            all_labels = []
            all_preds = []
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        acc.append(correct / total)
        cr = classification_report(all_labels, all_preds, output_dict=True)
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])

    # ... (Printing results remains the same)

if __name__ == '__main__':
    main()
