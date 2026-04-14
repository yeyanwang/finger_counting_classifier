"""
Evaluation Visualization Module
Generates Confusion Matrix, Robustness Decay, and Trade-off plots
according to the Evaluation Plan in the Proposal.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Ensure results directory exists
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Confusion Matrix (For Module 1 & 2)
def plot_confusion_matrix(y_true, y_pred, experiment_name):
    """
    Generates and saves a heatmap of the confusion matrix to diagnose class-specific errors.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
    
    plt.title(f'Confusion Matrix - {experiment_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    save_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{experiment_name.lower()}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

# 2. Robustness Decay Plot (For Module 2)
def plot_robustness_decay(ideal_acc, stressed_acc, model_name="PCA + K-NN"):
    """
    Plots a bar chart comparing performance on Ideal vs. Stressed datasets
    to visualize the environmental impact (decay in accuracy).
    """
    labels = ['Ideal Dataset', 'Stressed Dataset']
    accuracies = [ideal_acc, stressed_acc]
    decay = ideal_acc - stressed_acc
    
    plt.figure(figsize=(7, 6))
    bars = plt.bar(labels, accuracies, color=['#4C72B0', '#C44E52'], width=0.5)
    
    # Add accuracy text on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval*100:.1f}%', ha='center', va='bottom', fontsize=12)
        
    plt.ylim(0, 1.1)
    plt.ylabel('Test Accuracy')
    plt.title(f'Robustness Decay Analysis ({model_name})\nAccuracy Drop: {decay*100:.1f}%')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    save_path = os.path.join(RESULTS_DIR, f'robustness_decay_{model_name.replace(" ", "_").lower()}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Robustness Decay Plot saved to {save_path}")

# 3. Efficiency vs. Accuracy Trade-off Plot (For Module 2 & 3)
def plot_tradeoff(models_data):
    """
    Generates a scatter plot comparing accuracy against dimensionality
    to evaluate the balance between efficiency and performance.
    
    }
    """
    plt.figure(figsize=(9, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (name, stats), color in zip(models_data.items(), colors):
        plt.scatter(stats['dim'], stats['acc'], label=name, color=color, s=150, edgecolor='black', zorder=5)
        # Add slight offset for text label
        plt.text(stats['dim'] + 2, stats['acc'], name, fontsize=11, va='center')

    plt.title('Trade-off Plot: Efficiency vs. Accuracy')
    plt.xlabel('Number of Dimensions Used (Lower is more efficient)')
    plt.ylabel('Test Accuracy (Higher is better)')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    
    save_path = os.path.join(RESULTS_DIR, 'tradeoff_efficiency_vs_accuracy.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Trade-off Plot saved to {save_path}")
