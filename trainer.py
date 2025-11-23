import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import logging

import datasets
import ewc

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class TrainingHistory:
    def __init__(self):
        self.epoch_accuracies = []
        self.epoch_losses = []
        self.task_accuracies = []
        self.task_names = []
    
    def add_epoch(self, loss, accuracies):
        self.epoch_losses.append(loss)
        self.epoch_accuracies.append(accuracies)
    
    def add_task(self, task_name, accuracy):
        self.task_names.append(task_name)
        self.task_accuracies.append(accuracy)
    
    def get_accuracy_matrix(self):
        if not self.task_accuracies:
            return None
        return np.array(self.task_accuracies)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        logger.info(f"Early stopping initialized with patience={patience}, min_delta={min_delta}")
    
    def check(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            logger.debug(f"Initial validation loss: {val_loss:.4f}")
            return False
        
        if val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            logger.debug(f"Validation loss increased: {val_loss:.4f} > {self.best_loss:.4f} (counter: {self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered! Validation loss increased for {self.patience} consecutive epochs")
                return True
        else:
            if val_loss < self.best_loss:
                logger.debug(f"Validation loss improved: {val_loss:.4f} < {self.best_loss:.4f}")
                self.best_loss = val_loss
            self.counter = 0
        
        return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        logger.debug("Early stopping state reset")

def report(model, epoch, validation_datasets, batch_size, history=None):
    result = []
    accuracies = []
    total_loss = 0.0
    
    for inputs, labels in validation_datasets:
        loss, accuracy = model.evaluate(inputs, labels, verbose=0,
                                        batch_size=batch_size)
        result.append("{:.2f}".format(accuracy * 100))
        accuracies.append(accuracy)
        total_loss += loss
    
    avg_loss = total_loss / len(validation_datasets)
    
    if history is not None:
        history.add_epoch(avg_loss, accuracies)
    
    print(epoch + 1, "\t", "\t".join(result))
    return accuracies, avg_loss

def full_dataset(dataset_splits, increment):
    assert len(dataset_splits) == 1
    assert increment == 0
    return dataset_splits[increment]

def increment_dataset(dataset_splits, increment):
    return datasets.merge_data(dataset_splits[:increment + 1])

def switch_dataset(dataset_splits, increment):
    return dataset_splits[increment]

dataset_selector = {
    "full": full_dataset,
    "increment": increment_dataset,
    "switch": switch_dataset,
    "permute": switch_dataset
}

def increment_options():
    return sorted(dataset_selector.keys())

def compile_model(model, learning_rate, extra_losses=None):
    def custom_loss(y_true, y_pred):
        loss = sparse_categorical_crossentropy(y_true, y_pred)
        if extra_losses is not None:
            for fn in extra_losses:
                loss += fn(model)
        return loss

    model.compile(
        loss=custom_loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )

def train_epoch(model, train_data, batch_size,
                gradient_mask=None, incdet_threshold=None):
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(len(train_data[0])).batch(batch_size)

    for inputs, labels in dataset:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = model.compiled_loss(labels, outputs)

        gradients = tape.gradient(loss, model.trainable_weights)

        if gradient_mask is not None:
            gradients = ewc.apply_mask(gradients, gradient_mask)

        if incdet_threshold is not None:
            gradients = ewc.clip_gradients(gradients, incdet_threshold)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))

def train(model, train_data, valid_data, epochs, batch_size, learning_rate,
          dataset_update="full", increments=1,
          use_ewc=False, ewc_lambda=1, ewc_samples=100,
          use_fim=False, fim_threshold=1e-3, fim_samples=100,
          use_incdet=False, incdet_threshold=None,
          use_dropout=False, input_dropout=0.2, hidden_dropout=0.5,
          use_early_stopping=False, early_stopping_patience=5,
          plot_results=False, save_dir=None):
    
    history = TrainingHistory()
    
    early_stopper = EarlyStopping(patience=early_stopping_patience) if use_early_stopping else None
    
    compile_model(model, learning_rate)

    all_classes = np.unique(valid_data[1])
    class_sets = np.array_split(all_classes, increments)

    if dataset_update == "permute":
        train_sets, perms = datasets.permute_pixels(train_data, increments)
        valid_sets, _ = datasets.permute_pixels(valid_data, increments, perms)
    else:
        train_sets = datasets.split_data(train_data, classes=class_sets)
        valid_sets = datasets.split_data(valid_data, classes=class_sets)

    epochs_per_step = epochs // increments

    regularisers = []
    gradient_mask = None
    if not use_incdet:
        incdet_threshold = None

    for step in range(increments):
        inputs, labels = dataset_selector[dataset_update](train_sets, step)
        current_epoch = step * epochs_per_step

        if early_stopper:
            early_stopper.reset()

        for epoch in range(current_epoch, current_epoch + epochs_per_step):
            train_epoch(model, (inputs, labels), batch_size,
                        gradient_mask=gradient_mask,
                        incdet_threshold=incdet_threshold)

            accuracies, val_loss = report(model, epoch, valid_sets, batch_size, history)

            if early_stopper and early_stopper.check(val_loss):
                logger.info(f"Stopping training for current increment {step} at epoch {epoch}")
                break

        if use_ewc:
            loss_fn = ewc.ewc_loss(ewc_lambda, model, (inputs, labels),
                                   ewc_samples)
            regularisers.append(loss_fn)
            compile_model(model, learning_rate, extra_losses=regularisers)
        elif use_fim:
            new_mask = ewc.fim_mask(model, (inputs, labels), fim_samples,
                                    fim_threshold)
            gradient_mask = ewc.combine_masks(gradient_mask, new_mask)
    
    if plot_results:
        try:
            import evaluation
            import os
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            logger.info("Generating training visualizations")
            
            history_dict = {
                'loss': history.epoch_losses,
                'accuracy': [np.mean(accs) for accs in history.epoch_accuracies]
            }
            
            save_path = os.path.join(save_dir, 'training_history.png') if save_dir else None
            evaluation.plot_training_history(history_dict, 
                                            metrics=['loss', 'accuracy'],
                                            save_path=save_path)
            
            if increments > 1:
                accuracy_data = []
                for task_idx in range(increments):
                    task_accs = [accs[task_idx] if task_idx < len(accs) else 0.0 
                               for accs in history.epoch_accuracies]
                    accuracy_data.append(task_accs)
                
                save_path = os.path.join(save_dir, 'forgetting_curve.png') if save_dir else None
                evaluation.plot_forgetting_curve(accuracy_data, 
                                                task_names=[f'Task {i+1}' for i in range(increments)],
                                                save_path=save_path)
            
            logger.info("Visualizations generated successfully")
        except ImportError:
            logger.warning("evaluation module not available, skipping plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            
    return history