'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Calculate all metrics
        accuracy = accuracy_score(true_y, pred_y)
        precision = precision_score(true_y, pred_y, average='binary')
        recall = recall_score(true_y, pred_y, average='binary')
        f1 = f1_score(true_y, pred_y, average='binary')
        
        # Print all metrics
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F1 Score:', f1)
        
        # Return all metrics as a dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        