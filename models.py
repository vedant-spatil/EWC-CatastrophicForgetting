import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.activations import relu, softmax, tanh
import logging
import unittest
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

try:
    import datasets
    logger.info("Successfully imported datasets module")
except ImportError as e:
    logger.error(f"Failed to import datasets module: {e}")
    raise

class MLP(tf.keras.Model):

    def __init__(self, dataset=None, use_dropout=False, input_dropout=0.0, hidden_dropout=0.0):
        super(MLP, self).__init__()
        
        self.dataset = "mnist" if dataset is None else dataset
        logger.info(f"Initializing MLP for dataset: {self.dataset}")
        
        try:
            input_shape = datasets.input_shape(self.dataset)
            num_classes = datasets.num_classes(self.dataset)
            aug_pipeline = datasets.augmentations(self.dataset)
            
            logger.debug(f"MLP config - input_shape: {input_shape}, num_classes: {num_classes}")
            
            # Dynamic layer construction
            layers = [aug_pipeline, Flatten(input_shape=input_shape)]
            
            if use_dropout:
                layers.append(Dropout(input_dropout))
                
            layers.append(Dense(100, activation=relu))
            
            if use_dropout:
                layers.append(Dropout(hidden_dropout))
                
            layers.append(Dense(100, activation=relu))
            layers.append(Dense(num_classes, activation=softmax))
            
            self.model = tf.keras.Sequential(layers)
            
            logger.info(f"MLP initialized with {len(self.model.layers)} layers")
        except Exception as e:
            logger.error(f"Error initializing MLP: {e}")
            raise

    def get_config(self):
        config = {"dataset": self.dataset}
        logger.debug(f"MLP config: {config}")
        return config

    def call(self, inputs, training=None, mask=None):
        logger.debug(f"MLP forward pass - input shape: {inputs.shape}, training: {training}")
        return self.model.call(inputs, training, mask)

class LeNet5(tf.keras.Model):

    def __init__(self, dataset=None, use_dropout=False, input_dropout=0.0, hidden_dropout=0.0):
        super(LeNet5, self).__init__()
        
        self.dataset = "mnist" if dataset is None else dataset
        logger.info(f"Initializing LeNet5 for dataset: {self.dataset}")
        
        try:
            input_shape = datasets.input_shape(self.dataset)
            num_classes = datasets.num_classes(self.dataset)
            aug_pipeline = datasets.augmentations(self.dataset)
            
            logger.debug(f"LeNet5 config - input_shape: {input_shape}, num_classes: {num_classes}")
            
            layers = [aug_pipeline]

            if use_dropout:
                layers.append(Dropout(input_dropout))

            layers.extend([
                Conv2D(6, kernel_size=5, activation=tanh,
                       padding="same",
                       input_shape=input_shape),
                AveragePooling2D(pool_size=2),
                Conv2D(16, kernel_size=5, activation=tanh),
                AveragePooling2D(pool_size=2),
                Conv2D(120, kernel_size=5, activation=tanh),
                Flatten(),
                Dense(84, activation=tanh)
            ])

            if use_dropout:
                layers.append(Dropout(hidden_dropout))

            layers.append(Dense(num_classes, activation=softmax))
            
            self.model = tf.keras.Sequential(layers)
            
            logger.info(f"LeNet5 initialized with {len(self.model.layers)} layers")
        except Exception as e:
            logger.error(f"Error initializing LeNet5: {e}")
            raise

    def get_config(self):
        config = {"dataset": self.dataset}
        logger.debug(f"LeNet5 config: {config}")
        return config

    def call(self, inputs, training=None, mask=None):
        logger.debug(f"LeNet5 forward pass - input shape: {inputs.shape}, training: {training}")
        return self.model.call(inputs, training, mask)

class CifarNet(tf.keras.Model):

    def __init__(self, dataset=None, use_dropout=False, input_dropout=0.0, hidden_dropout=0.0):
        super(CifarNet, self).__init__()
        
        self.dataset = "cifar10" if dataset is None else dataset
        logger.info(f"Initializing CifarNet for dataset: {self.dataset}")
        
        # Default to 0.0 if use_dropout is False, otherwise use the provided hidden_dropout rate
        dropout_rate = hidden_dropout if use_dropout else 0.0
        
        try:
            input_shape = datasets.input_shape(self.dataset)
            self.num_classes = datasets.num_classes(self.dataset)
            aug_pipeline = datasets.augmentations(self.dataset)
            
            logger.debug(f"CifarNet config - input_shape: {input_shape}, num_classes: {self.num_classes}")
            
            self.model = tf.keras.Sequential([
                aug_pipeline,
                Conv2D(64, kernel_size=3, padding="same", activation=relu,
                       input_shape=input_shape),
                BatchNormalization(),
                Conv2D(64, kernel_size=3, padding="same", activation=relu),
                BatchNormalization(),
                Conv2D(128, kernel_size=3, padding="same", activation=relu,
                       strides=2),
                BatchNormalization(),
                Conv2D(128, kernel_size=3, padding="same", activation=relu),
                BatchNormalization(),
                Dropout(rate=dropout_rate), # Updated to use dynamic rate
                Conv2D(128, kernel_size=3, padding="same", activation=relu),
                BatchNormalization(),
                Conv2D(192, kernel_size=3, padding="same", activation=relu,
                       strides=2),
                BatchNormalization(),
                Conv2D(192, kernel_size=3, padding="same", activation=relu),
                BatchNormalization(),
                Dropout(rate=dropout_rate), # Updated to use dynamic rate
                Conv2D(192, kernel_size=3, padding="same", activation=relu),
                BatchNormalization(),
                AveragePooling2D(pool_size=8),
                Flatten(),
                Dense(self.num_classes, activation=softmax)
            ])
            
            logger.info(f"CifarNet initialized with {len(self.model.layers)} layers")
        except Exception as e:
            logger.error(f"Error initializing CifarNet: {e}")
            raise

    def get_config(self):
        config = {"dataset": self.dataset, "num_classes": self.num_classes}
        logger.debug(f"CifarNet config: {config}")
        return config

    def call(self, inputs, training=None, mask=None):
        logger.debug(f"CifarNet forward pass - input shape: {inputs.shape}, training: {training}")
        return self.model.call(inputs, training, mask)

model_dict = {
    "cifarnet": CifarNet,
    "lenet": LeNet5,
    "mlp": MLP
}

def models():
    logger.debug("Retrieving available model names")
    model_list = sorted(model_dict.keys())
    logger.info(f"Available models: {model_list}")
    return model_list

def get_model(name, dataset=None, use_dropout=False, input_dropout=0.0, hidden_dropout=0.0):
    logger.info(f"Creating model: {name} for dataset: {dataset}")
    
    if name not in model_dict:
        logger.error(f"Unknown model: {name}")
        raise ValueError(f"Unknown model: {name}. Available models: {list(model_dict.keys())}")
    
    try:
        # Pass dropout arguments to ALL models
        model = model_dict[name](dataset, 
                               use_dropout=use_dropout, 
                               input_dropout=input_dropout, 
                               hidden_dropout=hidden_dropout)
            
        logger.info(f"Successfully created {name} model")
        return model
    except Exception as e:
        logger.error(f"Error creating model {name}: {e}")
        raise

class TestModels(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logger.info("Setting up models test suite")
        np.random.seed(42)
        tf.random.set_seed(42)
        logger.info("Test setup completed")
    
    def test_models_returns_list(self):
        logger.info("Testing models() returns list")
        result = models()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        logger.info("models() list test passed")
    
    def test_models_contains_expected(self):
        logger.info("Testing models() contains expected models")
        result = models()
        expected = ["cifarnet", "lenet", "mlp"]
        for model_name in expected:
            self.assertIn(model_name, result)
        logger.info("Expected models test passed")
    
    def test_models_is_sorted(self):
        logger.info("Testing models() returns sorted list")
        result = models()
        self.assertEqual(result, sorted(result))
        logger.info("Sorted models test passed")
    
    def test_get_model_invalid_name(self):
        logger.info("Testing get_model() with invalid name")
        with self.assertRaises(ValueError):
            get_model("nonexistent_model")
        logger.info("Invalid model name test passed")
    
    def test_mlp_initialization_default(self):
        logger.info("Testing MLP initialization with default dataset")
        model = MLP()
        self.assertEqual(model.dataset, "mnist")
        self.assertIsInstance(model.model, tf.keras.Sequential)
        logger.info("MLP default initialization test passed")
    
    def test_mlp_initialization_with_dataset(self):
        logger.info("Testing MLP initialization with specified dataset")
        model = MLP(dataset="mnist")
        self.assertEqual(model.dataset, "mnist")
        logger.info("MLP dataset initialization test passed")
    
    def test_mlp_get_config(self):
        logger.info("Testing MLP get_config()")
        model = MLP(dataset="mnist")
        config = model.get_config()
        self.assertIn("dataset", config)
        self.assertEqual(config["dataset"], "mnist")
        logger.info("MLP config test passed")
    
    def test_mlp_forward_pass(self):
        logger.info("Testing MLP forward pass")
        model = MLP(dataset="mnist")
        dummy_input = np.random.rand(2, 28, 28, 1).astype(np.float32)
        output = model(dummy_input, training=False)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 10)
        
        probs_sum = tf.reduce_sum(output, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(probs_sum - 1.0) < 1e-5))
        
        logger.info("MLP forward pass test passed")
    
    def test_mlp_output_range(self):
        logger.info("Testing MLP output range")
        model = MLP(dataset="mnist")
        dummy_input = np.random.rand(5, 28, 28, 1).astype(np.float32)
        output = model(dummy_input, training=False)
        
        self.assertTrue(tf.reduce_all(output >= 0))
        self.assertTrue(tf.reduce_all(output <= 1))
        
        logger.info("MLP output range test passed")
    
    def test_lenet5_initialization_default(self):
        logger.info("Testing LeNet5 initialization with default dataset")
        model = LeNet5()
        self.assertEqual(model.dataset, "mnist")
        self.assertIsInstance(model.model, tf.keras.Sequential)
        logger.info("LeNet5 default initialization test passed")
    
    def test_lenet5_initialization_with_dataset(self):
        logger.info("Testing LeNet5 initialization with specified dataset")
        model = LeNet5(dataset="mnist")
        self.assertEqual(model.dataset, "mnist")
        logger.info("LeNet5 dataset initialization test passed")
    
    def test_lenet5_get_config(self):
        logger.info("Testing LeNet5 get_config()")
        model = LeNet5(dataset="mnist")
        config = model.get_config()
        self.assertIn("dataset", config)
        self.assertEqual(config["dataset"], "mnist")
        logger.info("LeNet5 config test passed")
    
    def test_lenet5_forward_pass(self):
        logger.info("Testing LeNet5 forward pass")
        model = LeNet5(dataset="mnist")
        dummy_input = np.random.rand(2, 28, 28, 1).astype(np.float32)
        output = model(dummy_input, training=False)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 10)
        
        probs_sum = tf.reduce_sum(output, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(probs_sum - 1.0) < 1e-5))
        
        logger.info("LeNet5 forward pass test passed")
    
    def test_lenet5_has_conv_layers(self):
        logger.info("Testing LeNet5 has convolutional layers")
        model = LeNet5(dataset="mnist")
        
        has_conv = any(isinstance(layer, Conv2D) for layer in model.model.layers)
        self.assertTrue(has_conv)
        
        logger.info("LeNet5 conv layers test passed")
    
    def test_cifarnet_initialization_default(self):
        logger.info("Testing CifarNet initialization with default dataset")
        model = CifarNet()
        self.assertEqual(model.dataset, "cifar10")
        self.assertEqual(model.num_classes, 10)
        self.assertIsInstance(model.model, tf.keras.Sequential)
        logger.info("CifarNet default initialization test passed")
    
    def test_cifarnet_initialization_cifar100(self):
        logger.info("Testing CifarNet initialization with CIFAR-100")
        model = CifarNet(dataset="cifar100")
        self.assertEqual(model.dataset, "cifar100")
        self.assertEqual(model.num_classes, 100)
        logger.info("CifarNet CIFAR-100 initialization test passed")
    
    def test_cifarnet_get_config(self):
        logger.info("Testing CifarNet get_config()")
        model = CifarNet(dataset="cifar10")
        config = model.get_config()
        self.assertIn("dataset", config)
        self.assertIn("num_classes", config)
        self.assertEqual(config["num_classes"], 10)
        logger.info("CifarNet config test passed")
    
    def test_cifarnet_forward_pass(self):
        logger.info("Testing CifarNet forward pass")
        model = CifarNet(dataset="cifar10")
        dummy_input = np.random.rand(2, 32, 32, 3).astype(np.float32)
        output = model(dummy_input, training=False)
        
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 10)
        
        probs_sum = tf.reduce_sum(output, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(probs_sum - 1.0) < 1e-5))
        
        logger.info("CifarNet forward pass test passed")
    
    def test_cifarnet_has_batchnorm(self):
        logger.info("Testing CifarNet has batch normalization")
        model = CifarNet(dataset="cifar10")
        
        has_bn = any(isinstance(layer, BatchNormalization) for layer in model.model.layers)
        self.assertTrue(has_bn)
        
        logger.info("CifarNet batch norm test passed")
    
    def test_cifarnet_has_dropout(self):
        logger.info("Testing CifarNet has dropout")
        model = CifarNet(dataset="cifar10")
        
        has_dropout = any(isinstance(layer, Dropout) for layer in model.model.layers)
        self.assertTrue(has_dropout)
        
        logger.info("CifarNet dropout test passed")
    
    def test_get_model_mlp(self):
        logger.info("Testing get_model() for MLP")
        model = get_model("mlp", dataset="mnist")
        self.assertIsInstance(model, MLP)
        self.assertEqual(model.dataset, "mnist")
        logger.info("get_model MLP test passed")
    
    def test_get_model_lenet(self):
        logger.info("Testing get_model() for LeNet5")
        model = get_model("lenet", dataset="mnist")
        self.assertIsInstance(model, LeNet5)
        self.assertEqual(model.dataset, "mnist")
        logger.info("get_model LeNet5 test passed")
    
    def test_get_model_cifarnet(self):
        logger.info("Testing get_model() for CifarNet")
        model = get_model("cifarnet", dataset="cifar10")
        self.assertIsInstance(model, CifarNet)
        self.assertEqual(model.dataset, "cifar10")
        logger.info("get_model CifarNet test passed")
    
    def test_get_model_default_dataset(self):
        logger.info("Testing get_model() with default dataset")
        mlp = get_model("mlp")
        lenet = get_model("lenet")
        cifarnet = get_model("cifarnet")
        
        self.assertEqual(mlp.dataset, "mnist")
        self.assertEqual(lenet.dataset, "mnist")
        self.assertEqual(cifarnet.dataset, "cifar10")
        
        logger.info("get_model default dataset test passed")
    
    def test_models_trainable(self):
        logger.info("Testing models are trainable")
        
        # MLP - need to call model first to build weights
        mlp = MLP(dataset="mnist")
        dummy_input_mnist = np.random.rand(1, 28, 28, 1).astype(np.float32)
        mlp(dummy_input_mnist, training=False)
        self.assertGreater(len(mlp.trainable_weights), 0)
        
        # LeNet5
        lenet = LeNet5(dataset="mnist")
        lenet(dummy_input_mnist, training=False)
        self.assertGreater(len(lenet.trainable_weights), 0)
        
        # CifarNet
        cifarnet = CifarNet(dataset="cifar10")
        dummy_input_cifar = np.random.rand(1, 32, 32, 3).astype(np.float32)
        cifarnet(dummy_input_cifar, training=False)
        self.assertGreater(len(cifarnet.trainable_weights), 0)
        
        logger.info("Models trainable test passed")
    
    def test_training_mode_affects_dropout(self):
        logger.info("Testing training mode affects dropout")
        tf.keras.backend.clear_session()  # Clear any stale graph state
        
        model = CifarNet(dataset="cifar10")
        dummy_input = np.random.rand(10, 32, 32, 3).astype(np.float32)
        
        output_train = model(dummy_input, training=True)
        output_test = model(dummy_input, training=False)
        
        outputs_different = not tf.reduce_all(tf.abs(output_train - output_test) < 1e-6)
        self.assertTrue(outputs_different)
        
        logger.info("Training mode dropout test passed")

def run_tests():
    logger.info("=" * 70)
    logger.info("Starting Models Test Suite")
    logger.info("=" * 70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModels)
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
