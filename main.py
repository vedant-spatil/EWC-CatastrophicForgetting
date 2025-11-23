import argparse
import textwrap
import logging
import sys

import datasets
import models
import trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def validate_args(args):
    logger.debug("Validating arguments")
    
    if args.dataset_update == "full":
        args.splits = 1
        logger.info("Dataset update set to 'full', forcing splits=1")
    
    if args.ewc and args.fim:
        logger.warning("Both EWC and FIM enabled - FIM will take precedence")
    
    if args.incdet and not args.ewc:
        logger.warning("IncDet enabled without EWC - IncDet has no effect")
    
    if args.splits < 1:
        logger.error(f"Invalid splits value: {args.splits}")
        raise ValueError("splits must be >= 1")
    
    if args.epochs < args.splits:
        logger.warning(f"epochs ({args.epochs}) < splits ({args.splits}), some tasks will have 0 epochs")
    
    return args


def main(args):
    logger.info("=" * 70)
    logger.info("INCREMENTAL LEARNING TRAINING SESSION")
    logger.info("=" * 70)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset if args.dataset else 'default for model'}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Dataset update: {args.dataset_update}, Splits: {args.splits}")
    
    if args.ewc:
        logger.info(f"EWC enabled - lambda: {args.ewc_lambda}, samples: {args.ewc_samples}")
    if args.fim:
        logger.info(f"FIM enabled - threshold: {args.fim_threshold}, samples: {args.fim_samples}")
    if args.incdet:
        logger.info(f"IncDet enabled - threshold: {args.incdet_threshold}")
    
    if args.dropout:
        logger.info(f"Dropout enabled - input: {args.input_dropout}, hidden: {args.hidden_dropout}")
    
    if args.early_stopping:
        logger.info(f"Early stopping enabled - patience: {args.patience} epochs")
    
    logger.info("Initializing model")
    model = models.get_model(args.model, args.dataset)
    dataset_name = model.dataset
    logger.info(f"Using dataset: {dataset_name}")
    
    logger.info("Loading dataset")
    train, test = datasets.load_dataset(dataset_name)
    logger.info(f"Train samples: {len(train[0])}, Test samples: {len(test[0])}")
    
    logger.info("Splitting train data into train/validation")
    train, validation = datasets.split_data(train, fractions=[0.8, 0.2])
    logger.info(f"Train: {len(train[0])}, Validation: {len(validation[0])}")
    
    logger.info("Starting training")
    try:
        history = trainer.train(
            model, train, validation, args.epochs, args.batch_size,
            args.learning_rate,
            dataset_update=args.dataset_update, increments=args.splits,
            use_ewc=args.ewc, ewc_lambda=args.ewc_lambda,
            ewc_samples=args.ewc_samples,
            use_fim=args.fim, fim_threshold=args.fim_threshold,
            fim_samples=args.fim_samples,
            use_incdet=args.incdet,
            incdet_threshold=args.incdet_threshold,
            use_dropout=args.dropout,
            input_dropout=args.input_dropout,
            hidden_dropout=args.hidden_dropout,
            use_early_stopping=args.early_stopping,
            early_stopping_patience=args.patience,
            plot_results=args.plot,
            save_dir=args.save_dir if args.plot else None
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    logger.info("Evaluating on test set")
    test_loss, test_accuracy = model.evaluate(test[0], test[1], 
                                               batch_size=args.batch_size, 
                                               verbose=0)
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")
    logger.info(f"Final test loss: {test_loss:.4f}")
    
    if args.evaluate:
        try:
            import evaluation
            import os
            
            logger.info("Running detailed evaluation")
            
            metrics = evaluation.evaluate_model(model, test[0], test[1], 
                                                batch_size=args.batch_size)
            
            evaluation.print_evaluation_report(metrics)
            
            if args.plot:
                os.makedirs(args.save_dir, exist_ok=True)
                
                logger.info(f"Saving evaluation plots to {args.save_dir}")
                
                cm_path = os.path.join(args.save_dir, 'confusion_matrix.png')
                evaluation.plot_confusion_matrix(metrics['confusion_matrix'], 
                                                save_path=cm_path,
                                                title=f'{args.model.upper()} - Confusion Matrix')
                
                pca_path = os.path.join(args.save_dir, 'per_class_accuracy.png')
                evaluation.plot_per_class_accuracy(metrics['per_class_accuracy'],
                                                  save_path=pca_path,
                                                  title=f'{args.model.upper()} - Per-Class Accuracy')
                
                logger.info("All plots saved successfully")
        
        except ImportError:
            logger.warning("evaluation module not available")
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
    
    logger.info("=" * 70)
    logger.info("TRAINING SESSION COMPLETE")
    logger.info("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""\
    Incremental learning demo.
    
    Train a model on a variable dataset. Dataset update options are:    
        1. full: Use the full dataset for the whole training process.
        2. increment: Start with a subset of classes and periodically add more.
        3. switch: Start with a subset of classes and periodically use a 
                   disjoint subset.
        4. permute: Start with the full dataset, and add permutations of the
                    pixels. Permutations are shared for each split.
    
    `splits` controls how many subsets to split the dataset into, with the given
    `epochs` being distributed evenly across them."""),
        formatter_class=argparse.RawTextHelpFormatter
    )

    training = parser.add_argument_group(title="Normal training")
    training.add_argument("--batch-size", type=int, default=256, metavar="N")
    training.add_argument("--epochs", type=int, default=30, metavar="N")
    training.add_argument("--learning-rate", type=float, default=0.001, metavar="LR")
    training.add_argument("--model", type=str, choices=models.models(), default="mlp")
    training.add_argument("--dataset", type=str, choices=datasets.datasets())

    inc = parser.add_argument_group(title="Incremental learning")
    inc.add_argument("--dataset-update", type=str,
                     choices=trainer.increment_options(), default="full")
    inc.add_argument("--splits", type=int, default=5, metavar="N")

    ewc_group = parser.add_argument_group(title="Elastic weight consolidation")
    ewc_group.add_argument("--ewc", action="store_true")
    ewc_group.add_argument("--ewc-lambda", type=float, default=0.1, metavar="L")
    ewc_group.add_argument("--ewc-samples", type=int, default=100, metavar="N")

    fim_group = parser.add_argument_group(title="Fisher information masking")
    fim_group.add_argument("--fim", action="store_true")
    fim_group.add_argument("--fim-threshold", type=float, default=1e-6, metavar="T")
    fim_group.add_argument("--fim-samples", type=int, default=100, metavar="N")

    incdet_group = parser.add_argument_group(title="Incremental detection")
    incdet_group.add_argument("--incdet", action="store_true")
    incdet_group.add_argument("--incdet-threshold", type=float, default=1e-6, metavar="B")

    regularization = parser.add_argument_group(title="Additional regularization")
    regularization.add_argument("--dropout", action="store_true",
                               help="Add dropout regularization to model")
    regularization.add_argument("--input-dropout", type=float, default=0.2,
                               help="Dropout rate for input layer")
    regularization.add_argument("--hidden-dropout", type=float, default=0.5,
                               help="Dropout rate for hidden layers")
    regularization.add_argument("--early-stopping", action="store_true",
                               help="Enable early stopping")
    regularization.add_argument("--patience", type=int, default=5,
                               help="Early stopping patience (epochs)")

    verbosity = parser.add_argument_group(title="Logging")
    verbosity.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          help="Set logging verbosity")
    verbosity.add_argument("--log-file", type=str, default="training.log",
                          help="Log file path")
    
    plotting = parser.add_argument_group(title="Visualization")
    plotting.add_argument("--plot", action="store_true",
                         help="Generate training plots after completion")
    plotting.add_argument("--save-dir", type=str, default="plots",
                         help="Directory to save plots and results")
    plotting.add_argument("--evaluate", action="store_true",
                         help="Run detailed evaluation on test set")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.baseFilename = args.log_file
    
    logger.info(f"Logging to {args.log_file}")
    
    try:
        args = validate_args(args)
        main(args)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)