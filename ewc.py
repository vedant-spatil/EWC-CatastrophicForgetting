from copy import deepcopy
from typing import List, Optional, Tuple
import logging
import numpy as np
import tensorflow as tf
import unittest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Fisher Formula (F_i) = E[(∂log P(y|x,θ) / ∂θ_i)²]
def fisher_matrix(model:tf.keras.Model, dataset:Tuple[np.ndarray, np.ndarray], num_samples:int, batch_size:int=1)-> List[tf.Tensor]:

    logger.info(f"Calculating Fisher Information Matrix {num_samples} samples, batch_size={batch_size}")
    
    input, labels=dataset
    weights=model.trainable_variables
    n_weights=len(weights)

    logger.debug(f"Model has {n_weights} trainable weight tensors")
    logger.debug(f"Dataset size: {len(input)} samples")

    variance=[tf.zeros_like(tensor) for tensor in weights]

    n_batches=(num_samples+batch_size-1)//batch_size

    logger.debug(f"Processing {n_batches} batches")

    all_indices = np.random.choice(input.shape[0], size=num_samples, replace=True)
    
    samples_processed = 0
    for batch_index in range(n_batches):
        batch_samples = min(batch_size, num_samples - batch_index * batch_size)
        
        batch_indices = all_indices[samples_processed:samples_processed + batch_samples]
        data = input[batch_indices]
        data = tf.convert_to_tensor(data, dtype=tf.float32)
        
        samples_processed += batch_samples

        with tf.GradientTape() as tape:
            prediction=model(data, training=False)
            log_likelihood = tf.reduce_sum(tf.math.log(prediction + 1e-10))
        
        gradients = tape.gradient(log_likelihood, weights)
        
        gradients = [
            tf.zeros_like(w) if g is None else g 
            for g, w in zip(gradients, weights)
        ]
        
        variance = [var + tf.reduce_sum(grad ** 2, axis=0, keepdims=False) for var, grad in zip(variance, gradients)]
        
        if (batch_index + 1) % max(1, n_batches // 10) == 0:
            logger.debug(f"Processed batch {batch_index + 1}/{n_batches}")

    fisher_diagonal = [tensor / num_samples for tensor in variance]
    
    for i, fisher in enumerate(fisher_diagonal):
        mean_val = tf.reduce_mean(fisher).numpy()
        max_val = tf.reduce_max(fisher).numpy()
        min_val = tf.reduce_min(fisher).numpy()
        logger.debug(f"Fisher layer {i}: mean={mean_val:.6f}, max={max_val:.6f}, min={min_val:.6f}")
    
    logger.info("Fisher matrix computation completed")
    return fisher_diagonal

#Loss_EWC = (λ/2) * Σ F_i * (θ_i - θ*_i)²
def ewc_loss(
    lam: float,
    model: tf.keras.Model,
    dataset: Tuple[np.ndarray, np.ndarray],
    num_samples: int,
    fisher_diagonal: Optional[List[tf.Tensor]] = None
) -> callable:
    logger.info(f"Creating EWC loss with lambda={lam}, num_samples={num_samples}")
    
    optimal_weights = [w.numpy().copy() for w in model.trainable_weights]
    logger.debug(f"Stored {len(optimal_weights)} optimal weight tensors")
    
    if fisher_diagonal is None:
        logger.info("Fisher diagonal not provided, computing...")
        fisher_diagonal = fisher_matrix(model, dataset, num_samples)
    else:
        logger.info("Using pre-computed Fisher diagonal")
    
    def loss_fn(new_model: tf.keras.Model) -> tf.Tensor:
        loss = tf.constant(0.0, dtype=tf.float32)
        current_weights = new_model.trainable_weights
        
        for fisher, current, optimal in zip(fisher_diagonal, current_weights, optimal_weights):
            optimal_tensor = tf.constant(optimal, dtype=current.dtype)
            weight_diff = current - optimal_tensor
            layer_loss = tf.reduce_sum(fisher * (weight_diff ** 2))
            loss += layer_loss
        
        total_loss = loss * (lam / 2.0)
        return total_loss
    
    logger.info("EWC loss function created")
    return loss_fn

def fim_mask(
    model: tf.keras.Model,
    dataset: Tuple[np.ndarray, np.ndarray],
    num_samples: int,
    threshold: float,
    fisher_diagonal: Optional[List[tf.Tensor]] = None
) -> List[tf.Tensor]:
    logger.info(f"Creating FIM mask with threshold={threshold}")
    
    if fisher_diagonal is None:
        logger.info("Computing Fisher diagonal for mask")
        fisher_diagonal = fisher_matrix(model, dataset, num_samples)
    else:
        logger.info("Using pre-computed Fisher diagonal")
    
    mask = [tf.less(fisher, threshold) for fisher in fisher_diagonal]
    
    for i, m in enumerate(mask):
        trainable_count = tf.reduce_sum(tf.cast(m, tf.int32)).numpy()
        total_count = tf.size(m).numpy()
        pct = 100.0 * trainable_count / total_count
        logger.info(f"Layer {i}: {trainable_count}/{total_count} ({pct:.1f}%) weights trainable")
    
    return mask

def combine_masks(
    mask1: Optional[List[tf.Tensor]],
    mask2: Optional[List[tf.Tensor]]
) -> Optional[List[tf.Tensor]]:
    logger.debug("Combining masks")
    
    if mask1 is None and mask2 is None:
        logger.debug("Both masks are None, returning None")
        return None
    if mask1 is None:
        logger.debug("Mask1 is None, returning mask2")
        return mask2
    if mask2 is None:
        logger.debug("Mask2 is None, returning mask1")
        return mask1
    
    combined = [tf.logical_and(m1, m2) for m1, m2 in zip(mask1, mask2)]
    
    for i, m in enumerate(combined):
        trainable = tf.reduce_sum(tf.cast(m, tf.int32)).numpy()
        total = tf.size(m).numpy()
        logger.debug(f"Combined layer {i}: {trainable}/{total} trainable")
    
    logger.info("Masks combined successfully")
    return combined

def apply_mask(
    gradients: List[tf.Tensor],
    mask: Optional[List[tf.Tensor]]
) -> List[tf.Tensor]:
    if mask is None:
        logger.debug("No mask provided, returning original gradients")
        return gradients
    
    logger.debug("Applying gradient mask")
    masked = [
        grad * tf.cast(m, grad.dtype) 
        for grad, m in zip(gradients, mask)
    ]
    
    for i, (orig, masked_grad, m) in enumerate(zip(gradients, masked, mask)):
        zeroed = tf.size(m).numpy() - tf.reduce_sum(tf.cast(m, tf.int32)).numpy()
        total = tf.size(m).numpy()
        logger.debug(f"Layer {i}: {zeroed}/{total} gradients zeroed ({100.0*zeroed/total:.1f}%)")
    
    return masked

def clip_gradients(
    gradients: List[tf.Tensor],
    threshold: float
) -> List[tf.Tensor]:
    logger.debug(f"Clipping gradients with threshold={threshold}")
    
    clipped = []
    for i, grad in enumerate(gradients):
        abs_grad = tf.abs(grad)
        scale = threshold / tf.maximum(threshold, abs_grad)
        clipped_grad = grad * scale
        
        max_before = tf.reduce_max(abs_grad).numpy()
        max_after = tf.reduce_max(tf.abs(clipped_grad)).numpy()
        n_clipped = tf.reduce_sum(tf.cast(abs_grad > threshold, tf.int32)).numpy()
        total = tf.size(grad).numpy()
        
        if n_clipped > 0:
            logger.debug(
                f"Layer {i}: {n_clipped}/{total} gradients clipped ({100.0*n_clipped/total:.1f}%), "
                f"max: {max_before:.6f} -> {max_after:.6f}"
            )
        else:
            logger.debug(f"Layer {i}: No gradients clipped (max: {max_before:.6f})")
        
        clipped.append(clipped_grad)
    
    return clipped

class TestEWC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logger.info("Setting up EWC test suite")
        
        cls.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        np.random.seed(42)
        cls.X_train = np.random.randn(100, 5).astype(np.float32)
        cls.y_train = np.random.randint(0, 3, 100).astype(np.int32)
        cls.dataset = (cls.X_train, cls.y_train)
        
        cls.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        cls.model.fit(cls.X_train, cls.y_train, epochs=5, verbose=0)
        
        logger.info("Test setup completed")
    
    def test_fisher_matrix_shape(self):
        logger.info("Testing Fisher matrix shape")
        
        fisher = fisher_matrix(self.model, self.dataset, num_samples=10)
        weights = self.model.trainable_weights
        
        self.assertEqual(len(fisher), len(weights))
        for f, w in zip(fisher, weights):
            self.assertEqual(f.shape, w.shape)
        
        logger.info("Fisher matrix shape test passed")
    
    def test_fisher_matrix_positive(self):
        logger.info("Testing Fisher matrix positivity")
        
        fisher = fisher_matrix(self.model, self.dataset, num_samples=10)
        
        for i, f in enumerate(fisher):
            min_val = tf.reduce_min(f).numpy()
            self.assertGreaterEqual(min_val, 0.0, f"Layer {i} has negative Fisher values")
        
        logger.info("Fisher matrix positivity test passed")
    
    def test_fisher_matrix_deterministic(self):
        logger.info("Testing Fisher matrix determinism")
        
        np.random.seed(42)
        tf.random.set_seed(42)
        fisher1 = fisher_matrix(self.model, self.dataset, num_samples=20, batch_size=1)
        
        np.random.seed(42)
        tf.random.set_seed(42)
        fisher2 = fisher_matrix(self.model, self.dataset, num_samples=20, batch_size=1)
        
        for i, (f1, f2) in enumerate(zip(fisher1, fisher2)):
            max_diff = tf.reduce_max(tf.abs(f1 - f2)).numpy()
            self.assertLess(max_diff, 1e-6, 
                f"Layer {i}: Fisher not deterministic: max_diff={max_diff:.10f}")
        
        logger.info("Fisher matrix determinism test passed")
    
    def test_ewc_loss_increases_on_weight_change(self):
        logger.info("Testing EWC loss behavior")
        
        loss_fn = ewc_loss(lam=1.0, model=self.model, dataset=self.dataset, num_samples=10)
        
        initial_loss = loss_fn(self.model).numpy()
        self.assertLess(initial_loss, 1e-2) 

        modified_model = tf.keras.models.clone_model(self.model)
        modified_model.set_weights(self.model.get_weights())
   
        for weight in modified_model.trainable_weights:
            weight.assign_add(tf.random.normal(weight.shape, stddev=0.5))
        
        modified_loss = loss_fn(modified_model).numpy()
        self.assertGreater(modified_loss, initial_loss)
        
        logger.info(f"EWC loss test passed (initial={initial_loss:.6f}, modified={modified_loss:.6f})")
    
    def test_ewc_loss_lambda_scaling(self):
        logger.info("Testing EWC loss lambda scaling")
        
        fisher = fisher_matrix(self.model, self.dataset, num_samples=10)
        
        loss_fn1 = ewc_loss(lam=1.0, model=self.model, dataset=self.dataset, 
                           num_samples=10, fisher_diagonal=fisher)
        loss_fn2 = ewc_loss(lam=2.0, model=self.model, dataset=self.dataset, 
                           num_samples=10, fisher_diagonal=fisher)
        
        modified_model = tf.keras.models.clone_model(self.model)
        modified_model.set_weights(self.model.get_weights())
        for weight in modified_model.trainable_weights:
            weight.assign_add(tf.random.normal(weight.shape, stddev=0.1))
        
        loss1 = loss_fn1(modified_model).numpy()
        loss2 = loss_fn2(modified_model).numpy()
        
        ratio = loss2 / loss1
        self.assertAlmostEqual(ratio, 2.0, places=5)
        
        logger.info(f"Lambda scaling test passed (ratio={ratio:.6f})")
    
    def test_fim_mask_thresholding(self):
        logger.info("Testing FIM mask thresholding")
        
        fisher = fisher_matrix(self.model, self.dataset, num_samples=10)
        threshold = 0.01
        
        mask = fim_mask(self.model, self.dataset, num_samples=10, 
                       threshold=threshold, fisher_diagonal=fisher)
        
        for f, m in zip(fisher, mask):
            masked_fisher = tf.boolean_mask(f, m)
            if tf.size(masked_fisher) > 0:
                max_masked = tf.reduce_max(masked_fisher).numpy()
                self.assertLess(max_masked, threshold)
            
            unmasked_fisher = tf.boolean_mask(f, tf.logical_not(m))
            if tf.size(unmasked_fisher) > 0:
                min_unmasked = tf.reduce_min(unmasked_fisher).numpy()
                self.assertGreaterEqual(min_unmasked, threshold)
        
        logger.info("FIM mask thresholding test passed")
    
    def test_combine_masks_none_handling(self):
        logger.info("Testing combine_masks None handling")
        
        mask1 = [tf.constant([True, False, True])]
        mask2 = [tf.constant([True, True, False])]
        
        self.assertIsNone(combine_masks(None, None))
        self.assertEqual(combine_masks(mask1, None), mask1)
        self.assertEqual(combine_masks(None, mask2), mask2)
        
        logger.info("Combine masks None handling test passed")
    
    def test_combine_masks_logic(self):
        logger.info("Testing combine_masks AND logic")
        
        mask1 = [tf.constant([True, False, True, False])]
        mask2 = [tf.constant([True, True, False, False])]
        
        expected = [tf.constant([True, False, False, False])]
        result = combine_masks(mask1, mask2)
        
        for r, e in zip(result, expected):
            self.assertTrue(tf.reduce_all(tf.equal(r, e)))
        
        logger.info("Combine masks AND logic test passed")
    
    def test_apply_mask_zeros_gradients(self):
        logger.info("Testing apply_mask gradient zeroing")
        
        gradients = [tf.constant([1.0, 2.0, 3.0, 4.0])]
        mask = [tf.constant([True, False, True, False])]
        
        masked = apply_mask(gradients, mask)
        expected = [tf.constant([1.0, 0.0, 3.0, 0.0])]
        
        for m, e in zip(masked, expected):
            self.assertTrue(tf.reduce_all(tf.equal(m, e)))
        
        logger.info("Apply mask test passed")
    
    def test_apply_mask_none(self):
        logger.info("Testing apply_mask with None")
        
        gradients = [tf.constant([1.0, 2.0, 3.0])]
        result = apply_mask(gradients, None)
        
        self.assertEqual(result, gradients)
        
        logger.info("Apply mask None test passed")
    
    def test_clip_gradients_threshold(self):
        logger.info("Testing gradient clipping")
        
        gradients = [tf.constant([0.5, 1.0, 2.0, 5.0, -3.0])]
        threshold = 2.0
        
        clipped = clip_gradients(gradients, threshold)
        
        for c in clipped:
            max_abs = tf.reduce_max(tf.abs(c)).numpy()
            self.assertLessEqual(max_abs, threshold + 1e-6)
        
        logger.info("Gradient clipping test passed")
    
    def test_clip_gradients_preserves_small(self):
        logger.info("Testing gradient clipping preserves small values")
        
        gradients = [tf.constant([0.1, 0.5, -0.3])]
        threshold = 2.0
        
        clipped = clip_gradients(gradients, threshold)
        
        for orig, clip in zip(gradients, clipped):
            self.assertTrue(tf.reduce_all(tf.equal(orig, clip)))
        
        logger.info("Gradient clipping small values test passed")
    
    def test_end_to_end_workflow(self):
        logger.info("Testing end-to-end EWC workflow")
        
        fisher = fisher_matrix(self.model, self.dataset, num_samples=20)
        
        loss_fn = ewc_loss(lam=0.5, model=self.model, dataset=self.dataset,
                          num_samples=20, fisher_diagonal=fisher)
        
        mask = fim_mask(self.model, self.dataset, num_samples=20, 
                       threshold=0.01, fisher_diagonal=fisher)
        
        with tf.GradientTape() as tape:
            predictions = self.model(self.X_train[:10], training=True)
            task_loss = tf.keras.losses.sparse_categorical_crossentropy(
                self.y_train[:10], predictions
            )
            ewc_penalty = loss_fn(self.model)
            total_loss = tf.reduce_mean(task_loss) + ewc_penalty
        
        gradients = tape.gradient(total_loss, self.model.trainable_weights)
        masked_grads = apply_mask(gradients, mask)
        clipped_grads = clip_gradients(masked_grads, threshold=1.0)
        
        self.assertEqual(len(clipped_grads), len(self.model.trainable_weights))
        
        logger.info("End-to-end workflow test passed")

def run_tests():
    logger.info("=" * 70)
    logger.info("Starting EWC Test Suite")
    logger.info("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEWC)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    logger.info("=" * 70)
    if result.wasSuccessful():
        logger.info("ALL TESTS PASSED")
    else:
        logger.error(f"{len(result.failures)} TEST(S) FAILED")
        logger.error(f"{len(result.errors)} TEST(S) HAD ERRORS")
    logger.info("=" * 70)
    
    return result
