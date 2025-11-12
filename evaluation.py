import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

def evaluate_model(device, model, testloader):
    
    all_predictions = []
    all_targets = []
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Get predictions
        _, predicted = outputs.max(1)
        
        # Store predictions and targets
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Calculate batch statistics
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
            
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
    
  
    # Store results
    results = {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
    }
    
    return results


def print_evaluation_results(results):
    print(f"  Accuracy:           {results['accuracy']:.2f}%")
    print(f"  Precision:  {results['precision']:.2f}%")
    print(f"  Recall (Macro):     {results['recall']:.2f}%")
    print(f"  F1 Score (Macro):   {results['f1']:.2f}%")
