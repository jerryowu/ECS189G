from local_code.base_class.method import method
from local_code.stage_1_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter
import re

class Method_RNN(method, nn.Module):
    data = None
    max_epoch = 50
    learning_rate = 1e-3
    max_vocab_size = 10000
    max_sequence_length = 200
    embedding_dim = 100
    hidden_size = 64
    num_layers = 2
    batch_size = 32

    loss_values = []

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.embedding = None
        self.rnn = None
        self.fc_layer = None
        self.activation_func_out = nn.Softmax(dim=1)
        self.loss_values = []
        self.is_generation_task = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def preprocess_text(self, texts):
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)

        word_counts = Counter(all_words)
        vocab = [word for word, _ in word_counts.most_common(self.max_vocab_size)]
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # 0 reserved for padding
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx) + 1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(input_size=self.embedding_dim,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         batch_first=True,
                         nonlinearity='tanh')  # tanh is the default activation for vanilla RNN
        self.fc_layer = nn.Linear(self.hidden_size, self.vocab_size if self.is_generation_task else 2)

    def text_to_sequence(self, text):
        words = re.findall(r'\b\w+\b', text.lower())
        sequence = [self.word_to_idx.get(word, 0) for word in words[:self.max_sequence_length]]
        if len(sequence) < self.max_sequence_length:
            sequence.extend([0] * (self.max_sequence_length - len(sequence)))
        return sequence

    def create_sequences(self, sequences, seq_length):
        X, y = [], []
        for seq in sequences:
            # Only create one sequence per input text
            if len(seq) > seq_length:
                X.append(seq[:seq_length])
                y.append(seq[seq_length])
        return X, y

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded, hidden)

        if self.is_generation_task:
            # Keep the sequence dimension
            output = self.fc_layer(rnn_out)  # Shape: [batch_size, seq_len, vocab_size]
            return output, hidden
        else:
            last_hidden = rnn_out[:, -1, :]
            x = self.fc_layer(last_hidden)
            y_pred = self.activation_func_out(x)
            return y_pred

    def train_model(self, X, y=None):
        if not self.word_to_idx:
            self.preprocess_text(X)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()

        if self.is_generation_task:
            sequences = [self.text_to_sequence(text) for text in X]
            seq_length = 20
            X_train, y_train = self.create_sequences(sequences, seq_length)
            
            # Create a DataLoader for batch processing
            X_tensor = torch.LongTensor(X_train)
            y_tensor = torch.LongTensor(y_train)
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.max_epoch):
                total_loss = 0
                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    output, _ = self.forward(batch_X)
                    # For generation task, we want to predict the next word
                    # So we take the last prediction for each sequence
                    output = output[:, -1, :]  # Shape: [batch_size, vocab_size]
                    loss = loss_function(output, batch_y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(loader)
                self.loss_values.append(avg_loss)
                print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}')
        else:
            accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

            X_sequences = [self.text_to_sequence(text) for text in X]
            X_tensor = torch.LongTensor(X_sequences)
            y_tensor = torch.LongTensor(np.array(y))
            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            for epoch in range(self.max_epoch):
                epoch_loss = 0
                all_preds = []
                all_true = []

                for batch_X, batch_y in loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    y_pred = self.forward(batch_X)
                    loss = loss_function(y_pred, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    all_preds.extend(y_pred.argmax(dim=1).cpu().tolist())
                    all_true.extend(batch_y.cpu().tolist())

                self.loss_values.append(epoch_loss / len(loader))
                if epoch % 10 == 0:
                    accuracy = accuracy_score(all_true, all_preds)
                    print(f"Epoch: {epoch}, Accuracy: {accuracy:.4f}, Loss: {epoch_loss / len(loader):.4f}")

    def test(self, X):
        self.eval()
        all_predictions = []
        
        # Create a DataLoader for batch processing
        X_sequences = [self.text_to_sequence(text) for text in X]
        X_tensor = torch.LongTensor(X_sequences)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_X, in loader:
                batch_X = batch_X.to(self.device)
                y_pred = self.forward(batch_X)
                predictions = y_pred.max(1)[1].cpu()
                all_predictions.extend(predictions.tolist())
        
        return torch.tensor(all_predictions)

    def generate_text(self, start_words, max_length=100):
        self.eval()
        with torch.no_grad():
            words = re.findall(r'\b\w+\b', ' '.join(start_words).lower())
            current_seq = [self.word_to_idx.get(word, 0) for word in words]

            # For vanilla RNN, we only need one hidden state
            h = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
            generated_words = words.copy()

            for _ in range(max_length):
                x = torch.LongTensor([current_seq[-20:]]).to(self.device)
                output, h = self.forward(x, h)
                # Get the last prediction and ensure it's a 1D tensor
                last_pred = output[0, -1]  # Shape: [vocab_size]
                probs = torch.softmax(last_pred, dim=0)
                # Sample from the probability distribution
                next_word_idx = torch.multinomial(probs, 1).item()
                current_seq.append(next_word_idx)
                generated_words.append(self.idx_to_word.get(next_word_idx, '<UNK>'))
                if next_word_idx == 0:
                    break

            return ' '.join(generated_words)

    def run(self):
        print('method running...')
        print('--start training...')

        self.is_generation_task = 'X' in self.data and 'y' not in self.data

        if self.is_generation_task:
            self.train_model(self.data['X'])
            print('--training completed...')
            return {'model': self}
        else:
            self.train_model(self.data['train']['X'], self.data['train']['y'])
            print('--start testing...')
            pred_y = self.test(self.data['test']['X'])
            self.plot_training_loss(self.loss_values)
            return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

    def plot_training_loss(self, loss_values):
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(loss_values)), loss_values, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Model Training Loss Over Epochs')
        plt.grid(True)
        plt.legend()
        plt.show()
