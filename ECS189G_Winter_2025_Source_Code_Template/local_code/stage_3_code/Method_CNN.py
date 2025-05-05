from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 28x28 → 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 14x14 → 7x7
        )

        # Fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Flatten(),  # Flatten the 3D feature maps to 1D
            nn.Linear(32 * 7 * 7, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Reshape input to [batch_size, channels, height, width]
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X)).view(-1, 1, 28, 28)
        y_tensor = torch.LongTensor(np.array(y))

        for epoch in range(self.max_epoch):
            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X)).view(-1, 1, 28, 28)
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
