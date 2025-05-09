'''
Concrete MethodModule class for a specific learning MethodModule (CNN version)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Define a simple CNN architecture
        self.conv_layer_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.activation_func_1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.activation_func_2 = nn.ReLU()

        # Assuming input images are 4x4 → after two conv + one 2x2 pool → output size 8 x 2 x 2
        self.fc_layer = nn.Linear(8 * 2 * 2, 2)
        self.activation_func_out = nn.Softmax(dim=1)

    def forward(self, x):
        '''Forward propagation'''
        x = self.activation_func_1(self.conv_layer_1(x))
        x = self.pool(x)
        x = self.activation_func_2(self.conv_layer_2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        y_pred = self.activation_func_out(x)
        return y_pred

    def train(self, X, y):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            # Convert X to tensor and ensure shape: (batch, channels=1, height, width)
            X_tensor = torch.FloatTensor(np.array(X))
            if X_tensor.dim() == 3:
                X_tensor = X_tensor.unsqueeze(1)  # Add channel dim if missing
            y_tensor = torch.LongTensor(np.array(y))

            y_pred = self.forward(X_tensor)
            train_loss = loss_function(y_pred, y_tensor)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    def test(self, X):
        X_tensor = torch.FloatTensor(np.array(X))
        if X_tensor.dim() == 3:
            X_tensor = X_tensor.unsqueeze(1)
        y_pred = self.forward(X_tensor)
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
