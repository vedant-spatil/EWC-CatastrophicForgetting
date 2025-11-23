import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import unittest

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def get_predictions(model, inputs, batch_size=256):
    predictions = model.predict(inputs, batch_size=batch_size, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    return predictions, predicted_classes

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted'):
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def calculate_recall(y_true, y_pred, average='weighted'):
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def calculate_f1(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def calculate_top_k_accuracy(y_true, y_probs, k=5):
    if y_probs.shape[1] < k:
        k = y_probs.shape[1]
    return top_k_accuracy_score(y_true, y_probs, k=k)

def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def calculate_per_class_accuracy(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    per_class_acc = {}
    for cls in range(num_classes):
        mask = y_true == cls
        if np.sum(mask) > 0:
            per_class_acc[cls] = np.mean(y_pred[mask] == cls)
        else:
            per_class_acc[cls] = 0.0
    return per_class_acc

def calculate_forgetting(accuracy_history):
    if len(accuracy_history) < 2:
        return 0.0
    
    max_acc = accuracy_history[0]
    forgetting = []
    
    for acc in accuracy_history[1:]:
        forgetting.append(max(0, max_acc - acc))
        max_acc = max(max_acc, acc)
    
    return np.mean(forgetting) if forgetting else 0.0

def calculate_backward_transfer(accuracy_matrix):
    n_tasks = accuracy_matrix.shape[0]
    if n_tasks < 2:
        return 0.0
    
    bwt = 0.0
    count = 0
    for i in range(n_tasks - 1):
        bwt += accuracy_matrix[-1, i] - accuracy_matrix[i, i]
        count += 1
    
    return bwt / count if count > 0 else 0.0

def calculate_forward_transfer(accuracy_matrix, random_baseline=None):
    n_tasks = accuracy_matrix.shape[0]
    if n_tasks < 2:
        return 0.0
    
    if random_baseline is None:
        random_baseline = np.zeros(n_tasks)
    
    fwt = 0.0
    count = 0
    for i in range(1, n_tasks):
        fwt += accuracy_matrix[i-1, i] - random_baseline[i]
        count += 1
    
    return fwt / count if count > 0 else 0.0

def evaluate_model(model, inputs, labels, batch_size=256, num_classes=None):
    logger.info("Starting model evaluation")
    
    y_probs, y_pred = get_predictions(model, inputs, batch_size)
    y_true = labels.flatten() if len(labels.shape) > 1 else labels
    
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1(y_true, y_pred),
        'top_5_accuracy': calculate_top_k_accuracy(y_true, y_probs, k=5),
        'confusion_matrix': calculate_confusion_matrix(y_true, y_pred),
        'per_class_accuracy': calculate_per_class_accuracy(y_true, y_pred, num_classes)
    }
    
    logger.info(f"Evaluation complete - Accuracy: {metrics['accuracy']:.4f}")
    return metrics

def evaluate_incremental(model, task_datasets, batch_size=256):
    logger.info("Starting incremental evaluation")
    
    n_tasks = len(task_datasets)
    accuracy_matrix = np.zeros((n_tasks, n_tasks))
    
    task_metrics = []
    for task_idx, (inputs, labels) in enumerate(task_datasets):
        metrics = evaluate_model(model, inputs, labels, batch_size)
        task_metrics.append(metrics)
        accuracy_matrix[-1, task_idx] = metrics['accuracy']
    
    results = {
        'task_metrics': task_metrics,
        'accuracy_matrix': accuracy_matrix,
        'average_accuracy': np.mean([m['accuracy'] for m in task_metrics]),
        'backward_transfer': calculate_backward_transfer(accuracy_matrix)
    }
    
    logger.info(f"Incremental evaluation complete - Avg Accuracy: {results['average_accuracy']:.4f}")
    return results

def print_evaluation_report(metrics, class_names=None):
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1 Score:       {metrics['f1_score']:.4f}")
    print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        name = class_names[cls] if class_names else f"Class {cls}"
        print(f"  {name}: {acc:.4f}")
    
    print("\n" + "=" * 50)

def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', 
                          save_path=None, figsize=(10, 8)):
    
    plt.figure(figsize=figsize)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_per_class_accuracy(per_class_acc, class_names=None, title='Per-Class Accuracy',
                            save_path=None, figsize=(12, 6)):
    
    plt.figure(figsize=figsize)
    
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    if class_names is None:
        class_names = [str(c) for c in classes]
    
    bars = plt.bar(class_names, accuracies, color='steelblue', edgecolor='black')
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.2f}')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Per-class accuracy plot saved to {save_path}")
    
    plt.show()

def plot_training_history(history, metrics=None, title='Training History',
                          save_path=None, figsize=(12, 5)):
    import matplotlib.pyplot as plt
    
    if metrics is None:
        metrics = ['accuracy', 'loss']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in history:
            ax.plot(history[metric], label=f'Train {metric}')
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Val {metric}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_accuracy_matrix(accuracy_matrix, title='Task Accuracy Matrix',
                         save_path=None, figsize=(8, 6)):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=figsize)
    
    n_tasks = accuracy_matrix.shape[0]
    task_labels = [f'Task {i+1}' for i in range(n_tasks)]
    
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=task_labels, yticklabels=[f'After {l}' for l in task_labels],
                vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel('Evaluated On')
    plt.ylabel('Trained After')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Accuracy matrix plot saved to {save_path}")
    
    plt.show()

def plot_forgetting_curve(accuracy_history_per_task, task_names=None,
                          title='Forgetting Curve', save_path=None, figsize=(10, 6)):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    n_tasks = len(accuracy_history_per_task)
    if task_names is None:
        task_names = [f'Task {i+1}' for i in range(n_tasks)]
    
    for i, (task_history, name) in enumerate(zip(accuracy_history_per_task, task_names)):
        epochs = range(len(task_history))
        plt.plot(epochs, task_history, marker='o', label=name, markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Forgetting curve plot saved to {save_path}")
    
    plt.show()

def plot_metrics_comparison(metrics_dict, title='Metrics Comparison',
                            save_path=None, figsize=(10, 6)):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    x = np.arange(len(metric_names))
    width = 0.8 / len(metrics_dict)
    
    for i, (label, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        bars = plt.bar(x + i * width, values, width, label=label)
        
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x + width * (len(metrics_dict) - 1) / 2, 
               [m.replace('_', ' ').title() for m in metric_names])
    plt.ylim(0, 1.15)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info(f"Metrics comparison plot saved to {save_path}")
    
    plt.show()

def plot_all_metrics(metrics, class_names=None, save_dir=None):
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, save_path=cm_path)
    
    pca_path = os.path.join(save_dir, 'per_class_accuracy.png') if save_dir else None
    plot_per_class_accuracy(metrics['per_class_accuracy'], class_names, save_path=pca_path)

class TestEvaluation(unittest.TestCase):   
    @classmethod
    def setUpClass(cls):
        logger.info("Setting up evaluation test suite")
        np.random.seed(42)
    
    def test_calculate_accuracy(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertEqual(calculate_accuracy(y_true, y_pred), 1.0)
        
        y_pred_wrong = np.array([0, 0, 0, 0, 0, 0])
        self.assertAlmostEqual(calculate_accuracy(y_true, y_pred_wrong), 2/6)
    
    def test_calculate_precision(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertEqual(calculate_precision(y_true, y_pred), 1.0)
    
    def test_calculate_recall(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertEqual(calculate_recall(y_true, y_pred), 1.0)
    
    def test_calculate_f1(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        self.assertEqual(calculate_f1(y_true, y_pred), 1.0)
    
    def test_calculate_confusion_matrix(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        cm = calculate_confusion_matrix(y_true, y_pred)
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        np.testing.assert_array_equal(cm, expected)
    
    def test_calculate_per_class_accuracy(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        per_class = calculate_per_class_accuracy(y_true, y_pred, num_classes=3)
        self.assertEqual(per_class[0], 0.5)
        self.assertEqual(per_class[1], 1.0)
        self.assertEqual(per_class[2], 0.5)
    
    def test_calculate_forgetting(self):
        history = [0.9, 0.85, 0.8, 0.75]
        forgetting = calculate_forgetting(history)
        self.assertGreater(forgetting, 0)
        
        history_no_forget = [0.7, 0.8, 0.85, 0.9]
        forgetting_none = calculate_forgetting(history_no_forget)
        self.assertEqual(forgetting_none, 0.0)
    
    def test_calculate_backward_transfer(self):
        acc_matrix = np.array([
            [0.9, 0.0, 0.0],
            [0.85, 0.9, 0.0],
            [0.8, 0.85, 0.9]
        ])
        bwt = calculate_backward_transfer(acc_matrix)
        expected = ((0.8 - 0.9) + (0.85 - 0.9)) / 2
        self.assertAlmostEqual(bwt, expected)
    
    def test_calculate_top_k_accuracy(self):
        y_true = np.array([0, 1, 2])
        y_probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7]
        ])
        top1 = calculate_top_k_accuracy(y_true, y_probs, k=1)
        self.assertEqual(top1, 1.0)
        
        top3 = calculate_top_k_accuracy(y_true, y_probs, k=3)
        self.assertEqual(top3, 1.0)
    
    def test_empty_history(self):
        self.assertEqual(calculate_forgetting([]), 0.0)
        self.assertEqual(calculate_forgetting([0.9]), 0.0)

def run_tests():
    logger.info("Starting Evaluation Test Suite")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEvaluation)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result
