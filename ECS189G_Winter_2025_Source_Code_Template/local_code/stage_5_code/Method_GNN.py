import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from local_code.base_class.method import method
from sklearn.metrics import f1_score, precision_score, recall_score

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        x = torch.spmm(adj, x)  # Sparse matrix multiply
        return self.linear(x)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, x, adj):
        x = self.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return x

class Method_GNN(method):
    data = None

    def __init__(self, mName, mDescription):
        super(Method_GNN, self).__init__(mName, mDescription)
        self.max_epoch = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden_dim = 16
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        print('method running...')
        print('--start training...')

        X = self.data['graph']['X']
        y = self.data['graph']['y']
        A = self.data['graph']['utility']['A']

        idx_train = self.data['train_test_val']['idx_train']
        idx_val = self.data['train_test_val']['idx_val']
        idx_test = self.data['train_test_val']['idx_test']

        model = GCN(X.shape[1], self.hidden_dim, y.max().item() + 1)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.train_losses = []
        val_accuracies = []

        for epoch in range(self.max_epoch):
            model.train()
            optimizer.zero_grad()

            out = model(X, A)
            loss = self.loss_fn(out[idx_train], y[idx_train])
            loss.backward()
            optimizer.step()

            acc, _, _, _ = self.test(model, X, A, y, idx_val)
            self.train_losses.append(loss.item())
            val_accuracies.append(acc)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Val Acc = {acc:.4f}")

        print('--start testing...')
        acc, f1, prec, rec = self.test(model, X, A, y, idx_test)
        print(f"Final Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}, Test Precision: {prec:.4f}, Test Recall: {rec:.4f}")
        self.plot_training_loss(self.train_losses)

    def test(self, model, X, A, y, idx):
        model.eval()
        with torch.no_grad():
            logits = model(X, A)
            preds = logits[idx].argmax(dim=1)
            true = y[idx].cpu().numpy()
            pred = preds.cpu().numpy()

            acc = (preds == y[idx]).sum().item() / len(idx)
            f1 = f1_score(true, pred, average='macro', zero_division=1)
            precision = precision_score(true, pred, average='macro', zero_division=1)
            recall = recall_score(true, pred, average='macro', zero_division=1)

            return acc, f1, precision, recall


    def plot_training_loss(self, losses):
        plt.figure()
        plt.plot(losses, label='Training Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.title("GCN Learning Curve")
        plt.show()
