import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RandomTranslation, RandomFlip, Normalization, RandomRotation
import logging
import unittest

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

dataset_metadata = {
    "mnist": {
        "dataset": tf.keras.datasets.mnist,
        "input_shape": (28, 28, 1),
        "num_classes": 10,
        "augmentations": [
            RandomTranslation(0.1, 0.1),
        ]
    },
    "cifar10": {
        "dataset": tf.keras.datasets.cifar10,
        "input_shape": (32, 32, 3),
        "num_classes": 10,
        "augmentations": [
            RandomTranslation(0.1, 0.1),
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            Normalization(mean=[0.4914, 0.4822, 0.4465],
                         variance=[0.04093, 0.03976, 0.04040]),
        ]
    },
    "cifar100": {
        "dataset": tf.keras.datasets.cifar100,
        "input_shape": (32, 32, 3),
        "num_classes": 100,
        "augmentations": [
            RandomTranslation(0.1, 0.1),
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            Normalization(mean=[0.5071, 0.4867, 0.4408],
                         variance=[0.0637, 0.0611, 0.0676])
        ]
    },
}

def datasets():
    logger.debug("Retrieving available dataset names")
    dataset_list = list(dataset_metadata.keys())
    logger.info(f"Available datasets: {dataset_list}")
    return dataset_list

def input_shape(dataset_name):
    logger.debug(f"Getting input shape for dataset: {dataset_name}")
    if dataset_name not in dataset_metadata:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    shape = dataset_metadata[dataset_name]["input_shape"]
    logger.info(f"Input shape for {dataset_name}: {shape}")
    return shape

def num_classes(dataset_name):
    logger.debug(f"Getting number of classes for dataset: {dataset_name}")
    if dataset_name not in dataset_metadata:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    n_classes = dataset_metadata[dataset_name]["num_classes"]
    logger.info(f"Number of classes for {dataset_name}: {n_classes}")
    return n_classes

def augmentations(dataset_name):
    logger.debug(f"Creating augmentation pipeline for dataset: {dataset_name}")
    if dataset_name not in dataset_metadata:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if dataset_name == "mnist":
        layers = [RandomTranslation(0.1, 0.1)]
    elif dataset_name == "cifar10":
        mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
        std = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        layers = [
            RandomTranslation(0.1, 0.1),
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            tf.keras.layers.Lambda(lambda x: (x - mean) / std),
        ]
    elif dataset_name == "cifar100":
        mean = tf.constant([0.5071, 0.4867, 0.4408], dtype=tf.float32)
        std = tf.constant([0.2675, 0.2565, 0.2761], dtype=tf.float32)
        layers = [
            RandomTranslation(0.1, 0.1),
            RandomFlip("horizontal"),
            RandomRotation(0.1),
            tf.keras.layers.Lambda(lambda x: (x - mean) / std),
        ]
    else:
        layers = []
    
    logger.info(f"Augmentation pipeline for {dataset_name}: {len(layers)} layers")
    return tf.keras.Sequential(layers)

def load_dataset(dataset_name):
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name not in dataset_metadata:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    dataset_loader = dataset_metadata[dataset_name]["dataset"]
    (x_train, y_train), (x_test, y_test) = dataset_loader.load_data()
    
    logger.debug(f"Raw data shapes - Train: {x_train.shape}, Test: {x_test.shape}")

    if dataset_name == "mnist":
        logger.debug("Expanding MNIST dimensions to add channel")
        x_train = np.expand_dims(x_train, 3)
        x_test = np.expand_dims(x_test, 3)
    
    y_train = y_train.reshape([-1])
    y_test = y_test.reshape([-1])
    
    x_train_normalized = x_train / 255.0
    x_test_normalized = x_test / 255.0
    
    logger.info(f"Dataset loaded - Train: {x_train_normalized.shape}, Test: {x_test_normalized.shape}")
    logger.debug(f"Train labels range: [{y_train.min()}, {y_train.max()}]")
    logger.debug(f"Test labels range: [{y_test.min()}, {y_test.max()}]")

    return (x_train_normalized, y_train), (x_test_normalized, y_test)

def split_data(data, fractions=None, classes=None):
    logger.info("Splitting data")
    
    if fractions is not None and classes is not None:
        logger.error("Cannot split by both fractions and classes simultaneously")
        raise RuntimeError("split_data can't split by fractions and classes simultaneously")

    inputs, labels = data
    logger.debug(f"Input data shape: {inputs.shape}, labels shape: {labels.shape}")

    if fractions is not None:
        logger.info(f"Splitting by fractions: {fractions}")
        
        permutation = np.random.permutation(len(inputs))
        inputs = inputs[permutation]
        labels = labels[permutation]
        logger.debug("Data permuted")

        cumulative = np.cumsum(fractions)
        indices = [int(fraction * len(inputs)) for fraction in cumulative]
        logger.debug(f"Split indices: {indices}")
        
        inputs = np.array_split(inputs, indices)
        labels = np.array_split(labels, indices)

        data = list(zip(inputs[:len(fractions)], labels[:len(fractions)]))
        
        for i, (inp, lbl) in enumerate(data):
            logger.info(f"Split {i}: {inp.shape[0]} samples ({100*fractions[i]:.1f}%)")
        
    elif classes is not None:
        logger.info(f"Splitting by classes: {classes}")
        
        masks = [np.isin(labels, class_list) for class_list in classes]
        inputs = [inputs[mask] for mask in masks]
        labels = [labels[mask] for mask in masks]

        data = list(zip(inputs, labels))
        
        for i, (inp, lbl) in enumerate(data):
            unique_classes = np.unique(lbl)
            logger.info(f"Split {i}: {inp.shape[0]} samples, classes: {unique_classes.tolist()}")
    else:
        logger.warning("No split criteria provided, returning original data")
        data = [data]

    return data

def merge_data(data):
    logger.info(f"Merging {len(data)} datasets")
    
    if len(data) == 0:
        logger.error("Cannot merge empty dataset list")
        raise ValueError("Cannot merge empty dataset list")
    
    inputs = [dataset[0] for dataset in data]
    labels = [dataset[1] for dataset in data]
    
    for i, (inp, lbl) in enumerate(zip(inputs, labels)):
        logger.debug(f"Dataset {i}: {inp.shape[0]} samples")
    
    merged_inputs = np.concatenate(inputs)
    merged_labels = np.concatenate(labels)
    
    logger.info(f"Merged result: {merged_inputs.shape[0]} total samples")
    
    return merged_inputs, merged_labels

def permute_pixels(data, permutations, permutation=None):
    logger.info(f"Creating {permutations} pixel permutations")
    
    result = [data]
    images, labels = data
    shape = images.shape
    pixels = shape[1] * shape[2]
    
    logger.debug(f"Image shape: {shape}, total pixels: {pixels}")

    flattened = np.reshape(images, [shape[0], pixels, -1])
    logger.debug(f"Flattened shape: {flattened.shape}")

    if permutation is None:
        logger.debug("Generating new permutations")
        permutation = []
        for i in range(1, permutations):
            perm = np.random.permutation(pixels)
            permutation.append(perm)
            logger.debug(f"Generated permutation {i}/{permutations-1}")
    else:
        logger.debug("Using provided permutations")

    assert len(permutation) == (permutations - 1), \
        f"Expected {permutations-1} permutations, got {len(permutation)}"

    for i, p in enumerate(permutation):
        permuted = flattened.copy()[:, p]
        permuted = np.reshape(permuted, shape)
        result.append((permuted, labels))
        logger.debug(f"Applied permutation {i+1}/{len(permutation)}")
    
    logger.info(f"Created {len(result)} permuted datasets")

    return result, permutation

class TestDatasetUtils(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.info("Setting up dataset utils test suite")
        np.random.seed(42)
        tf.random.set_seed(42)
        
        cls.sample_data = (
            np.random.rand(100, 28, 28, 1).astype(np.float32),
            np.random.randint(0, 10, 100)
        )
        cls.sample_rgb_data = (
            np.random.rand(50, 32, 32, 3).astype(np.float32),
            np.random.randint(0, 10, 50)
        )
        logger.info("Test setup completed")
    
    def test_datasets_returns_list(self):
        logger.info("Testing datasets() returns list")
        result = datasets()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        logger.info("datasets() test passed")
    
    def test_datasets_contains_expected(self):
        logger.info("Testing datasets() contains expected datasets")
        result = datasets()
        expected = ["mnist", "cifar10", "cifar100"]
        for dataset in expected:
            self.assertIn(dataset, result)
        logger.info("Expected datasets test passed")
    
    def test_input_shape_mnist(self):
        logger.info("Testing input_shape() for MNIST")
        shape = input_shape("mnist")
        self.assertEqual(shape, (28, 28, 1))
        logger.info("MNIST input shape test passed")
    
    def test_input_shape_cifar10(self):
        logger.info("Testing input_shape() for CIFAR-10")
        shape = input_shape("cifar10")
        self.assertEqual(shape, (32, 32, 3))
        logger.info("CIFAR-10 input shape test passed")
    
    def test_input_shape_invalid(self):
        logger.info("Testing input_shape() with invalid dataset")
        with self.assertRaises(ValueError):
            input_shape("invalid_dataset")
        logger.info("Invalid dataset test passed")
    
    def test_num_classes_mnist(self):
        logger.info("Testing num_classes() for MNIST")
        n = num_classes("mnist")
        self.assertEqual(n, 10)
        logger.info("MNIST num_classes test passed")
    
    def test_num_classes_cifar100(self):
        logger.info("Testing num_classes() for CIFAR-100")
        n = num_classes("cifar100")
        self.assertEqual(n, 100)
        logger.info("CIFAR-100 num_classes test passed")
    
    def test_num_classes_invalid(self):
        logger.info("Testing num_classes() with invalid dataset")
        with self.assertRaises(ValueError):
            num_classes("nonexistent")
        logger.info("Invalid num_classes test passed")
    
    def test_augmentations_returns_sequential(self):
        logger.info("Testing augmentations() returns Sequential model")
        aug = augmentations("mnist")
        self.assertIsInstance(aug, tf.keras.Sequential)
        logger.info("Augmentations Sequential test passed")
    
    def test_augmentations_has_layers(self):
        logger.info("Testing augmentations() has layers")
        aug = augmentations("cifar10")
        self.assertGreater(len(aug.layers), 0)
        logger.info(f"CIFAR-10 has {len(aug.layers)} augmentation layers")
    
    def test_augmentations_invalid(self):
        logger.info("Testing augmentations() with invalid dataset")
        with self.assertRaises(ValueError):
            augmentations("fake_dataset")
        logger.info("Invalid augmentations test passed")
    
    def test_split_data_by_fractions(self):
        logger.info("Testing split_data() by fractions")
        fractions = [0.6, 0.2, 0.2]
        splits = split_data(self.sample_data, fractions=fractions)
        
        self.assertEqual(len(splits), 3)
        
        total_samples = sum(split[0].shape[0] for split in splits)
        self.assertEqual(total_samples, 100)
        
        expected_sizes = [60, 20, 20]
        for i, (split, expected) in enumerate(zip(splits, expected_sizes)):
            self.assertEqual(split[0].shape[0], expected)
        
        logger.info("Split by fractions test passed")
    
    def test_split_data_by_classes(self):
        logger.info("Testing split_data() by classes")
        classes = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]
        splits = split_data(self.sample_data, classes=classes)
        
        self.assertEqual(len(splits), 3)
        
        for i, (split, class_list) in enumerate(zip(splits, classes)):
            unique_labels = np.unique(split[1])
            for label in unique_labels:
                self.assertIn(label, class_list)
        
        logger.info("Split by classes test passed")
    
    def test_split_data_both_params_raises(self):
        logger.info("Testing split_data() raises with both parameters")
        with self.assertRaises(RuntimeError):
            split_data(self.sample_data, fractions=[0.5, 0.5], classes=[[0, 1], [2, 3]])
        logger.info("Both parameters error test passed")
    
    def test_split_data_preserves_shape(self):
        logger.info("Testing split_data() preserves data shape")
        splits = split_data(self.sample_data, fractions=[0.5, 0.5])
        
        for split in splits:
            self.assertEqual(split[0].shape[1:], (28, 28, 1))
        
        logger.info("Shape preservation test passed")
    
    def test_merge_data_combines_correctly(self):
        logger.info("Testing merge_data() combines datasets")
        data1 = (np.random.rand(30, 28, 28, 1).astype(np.float32), np.array([0]*30))
        data2 = (np.random.rand(20, 28, 28, 1).astype(np.float32), np.array([1]*20))
        
        merged_inputs, merged_labels = merge_data([data1, data2])
        
        self.assertEqual(merged_inputs.shape[0], 50)
        self.assertEqual(merged_labels.shape[0], 50)
        self.assertEqual(merged_inputs.shape[1:], (28, 28, 1))
        
        logger.info("Merge data test passed")
    
    def test_merge_data_empty_raises(self):
        logger.info("Testing merge_data() raises on empty list")
        with self.assertRaises(ValueError):
            merge_data([])
        logger.info("Empty merge error test passed")
    
    def test_merge_data_preserves_order(self):
        logger.info("Testing merge_data() preserves order")
        data1 = (np.ones((10, 28, 28, 1)), np.zeros(10))
        data2 = (np.zeros((10, 28, 28, 1)), np.ones(10))
        
        merged_inputs, merged_labels = merge_data([data1, data2])
        
        self.assertTrue(np.all(merged_inputs[:10] == 1))
        self.assertTrue(np.all(merged_inputs[10:] == 0))
        self.assertTrue(np.all(merged_labels[:10] == 0))
        self.assertTrue(np.all(merged_labels[10:] == 1))
        
        logger.info("Order preservation test passed")
    
    def test_permute_pixels_creates_correct_number(self):
        logger.info("Testing permute_pixels() creates correct number")
        small_data = (np.random.rand(10, 28, 28, 1).astype(np.float32), np.arange(10))
        
        result, perms = permute_pixels(small_data, permutations=4)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(len(perms), 3)
        
        logger.info("Permute number test passed")
    
    def test_permute_pixels_preserves_shape(self):
        logger.info("Testing permute_pixels() preserves shape")
        small_data = (np.random.rand(10, 28, 28, 1).astype(np.float32), np.arange(10))
        
        result, _ = permute_pixels(small_data, permutations=3)
        
        for permuted_data, labels in result:
            self.assertEqual(permuted_data.shape, (10, 28, 28, 1))
            self.assertEqual(labels.shape, (10,))
        
        logger.info("Permute shape test passed")
    
    def test_permute_pixels_preserves_labels(self):
        logger.info("Testing permute_pixels() preserves labels")
        small_data = (np.random.rand(10, 28, 28, 1).astype(np.float32), np.arange(10))
        
        result, _ = permute_pixels(small_data, permutations=3)
        
        for _, labels in result:
            self.assertTrue(np.array_equal(labels, np.arange(10)))
        
        logger.info("Permute labels test passed")
    
    def test_permute_pixels_uses_provided_permutation(self):
        logger.info("Testing permute_pixels() uses provided permutation")
        small_data = (np.random.rand(5, 4, 4, 1).astype(np.float32), np.arange(5))
        
        perm1 = np.random.permutation(16)
        result1, _ = permute_pixels(small_data, permutations=2, permutation=[perm1])
        result2, _ = permute_pixels(small_data, permutations=2, permutation=[perm1])
        
        self.assertTrue(np.array_equal(result1[1][0], result2[1][0]))
        
        logger.info("Provided permutation test passed")
    
    def test_permute_pixels_changes_pixels(self):
        logger.info("Testing permute_pixels() actually changes pixels")
        small_data = (np.random.rand(5, 4, 4, 1).astype(np.float32), np.arange(5))
        
        result, _ = permute_pixels(small_data, permutations=2)
        
        original = result[0][0]
        permuted = result[1][0]
        
        self.assertFalse(np.array_equal(original, permuted))
        
        original_flat = original.reshape(5, -1)
        permuted_flat = permuted.reshape(5, -1)
        self.assertTrue(np.allclose(np.sort(original_flat), np.sort(permuted_flat)))
        
        logger.info("Pixel permutation test passed")
    
    def test_permute_pixels_deterministic_with_seed(self):
        logger.info("Testing permute_pixels() determinism")
        small_data = (np.random.rand(5, 4, 4, 1).astype(np.float32), np.arange(5))
        
        np.random.seed(123)
        result1, perm1 = permute_pixels(small_data, permutations=2)
        
        np.random.seed(123)
        result2, perm2 = permute_pixels(small_data, permutations=2)
        
        self.assertTrue(np.array_equal(perm1[0], perm2[0]))
        self.assertTrue(np.array_equal(result1[1][0], result2[1][0]))
        
        logger.info("Determinism test passed")

def run_tests():
    logger.info("=" * 70)
    logger.info("Starting Dataset Utils Test Suite")
    logger.info("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDatasetUtils)
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
