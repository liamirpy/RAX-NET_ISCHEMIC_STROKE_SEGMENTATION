from load_data import load_data
from statistic import Statistics
from compiler import ModelTrainer
from model import RAXNet
import tensorflow as tf
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

stats = Statistics()

def parameters_and_train(
        # Model Design
        image_size=(208, 240),
        num_classes=1,

        # Data Directory
        folder_directory='',
        train_mri='',
        train_mask='',
        validation_mri='',
        validation_mask='',

        # Learning Rate
        learning_rate=1e-2,

        # Loss Function ['FocalTverskyLoss', 'FocalLoss', 'TverskyLoss', 'DiceCoefLoss']
        loss_function='FocalTverskyLoss',

        # Metrics names ['accuracy', 'hausdorff_distance', 'dice_coef', 'recall', 'precision']
        metrics_=['dice_coef'],

        epochs=10,

        patients=20,

        checkpoint_path='',

        checkpoint_name=''):

    # LOAD MODEL
    rax_net = RAXNet(image_size, num_classes)
    model = rax_net.build_model()
    model.summary()

    # LOAD DATA
    train_data_mri, train_data_mask, validation_data_mri, validation_data_mask = load_data(
        folder_directory,
        train_mri,
        train_mask,
        validation_mri,
        validation_mask)

    # OPTIMIZER
    optimizer = stats.ADAMW(learning_rate)

    # LOSS FUNCTION
    loss_functions = {
        'FocalTverskyLoss': stats.FocalTverskyLoss,
        'FocalLoss': stats.FocalLoss,
        'TverskyLoss': stats.TverskyLoss,
        'DiceCoefLoss': stats.DiceCoefLoss
    }

    loss = loss_functions.get(loss_function, stats.FocalTverskyLoss)

    # METRICS
    valid_metrics = ['hausdorff_distance', 'dice_coef', 'recall', 'precision']
    metrics = []
    for metric in metrics_:
        if metric not in valid_metrics:
            raise ValueError("Invalid metric. Valid metrics are: {}".format(valid_metrics))
        metrics.append(getattr(stats, metric))


    trainer = ModelTrainer(
        model=model,

        train_data_mri=train_data_mri,

        train_data_mask=train_data_mask,

        validation_data_mri=validation_data_mri,

        validation_data_mask=validation_data_mask,

        optimizer=optimizer,

        loss=loss,

        metrics=metrics,

        metrics_names=['loss'] + metrics_,

        checkpoint_path=checkpoint_path,

        checkpoint_name=checkpoint_name,

        patience=1000,

        num_epochs=epochs
    )

    trainer.train()





# parameters_and_train(
#     image_size=(208, 240),

#     num_classes=1,

#     folder_directory='../Data_Preparation/Axial/fold_3',

#     train_mri='train_data_mri_fold_3.npy',

#     train_mask='train_data_mask_fold_3.npy',

#     validation_mri='validation_data_mri_fold_3.npy',

#     validation_mask='validation_data_mask_fold_3.npy',
#         # Learning Rate
#     learning_rate=1e-3,

#     # Loss Function ['FocalTverskyLoss', 'FocalLoss', 'TverskyLoss', 'DiceCoefLoss']
#     loss_function='FocalTverskyLoss',

#     # Metrics names ['accuracy', 'hausdorff_distance', 'dice_coef', 'recall', 'precision']
#     metrics_=['dice_coef','recall', 'precision'],

#     epochs=10,

#     patients=20,

#     save_result_in='./',

#     name='amir'

# )



import argparse


parser = argparse.ArgumentParser(description='Model Training Parameters')  
# Model Design
parser.add_argument('--image_size', nargs='+', type=int, default=[208, 240], help='Image size (height, width)')

parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')

# Data Directory
parser.add_argument('--folder_directory', type=str, help='Folder directory')

parser.add_argument('--train_mri', type=str, help='Training  data file name(.npy)')

parser.add_argument('--train_mask', type=str, help='Training mask data file name(.npy)')

parser.add_argument('--validation_mri', type=str, help='Validation data file name(.npy)')

parser.add_argument('--validation_mask', type=str, help='Validation mask data file name(.npy)')

# Learning Rate
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')

# Loss Function
parser.add_argument('--loss_function', type=str, default='DiceCoefLoss', choices=['FocalTverskyLoss', 'FocalLoss', 'TverskyLoss', 'DiceCoefLoss'], help='Loss function')

# Metrics names
parser.add_argument('--metrics_names', nargs='+', type=str, default=['dice_coef'], choices=['precision', 'recall', 'accuracy', 'hausdorff_distance','dice_coef'], help='Metrics names')

# Other parameters
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')

parser.add_argument('--patients', type=int, default=5, help='Number of patients')

parser.add_argument('--checkpoint_path', type=str, default='', help='Checkpoint path')

parser.add_argument('--checkpoint_name', type=str, default='', help='Checkpoint name (.h5 and csv)')

args=parser.parse_args()





parameters_and_train(
    image_size = tuple(args.image_size),

    num_classes = args.num_classes,

    folder_directory = args.folder_directory,

    train_mri = args.train_mri,

    train_mask = args.train_mask,

    validation_mri = args.validation_mri,

    validation_mask = args.validation_mask,

    learning_rate = args.learning_rate,

    loss_function = args.loss_function,

    metrics_ = args.metrics_names,

    epochs = args.epochs,

    patients = args.patients,

    checkpoint_path = args.checkpoint_path,

    checkpoint_name = args.checkpoint_name)


# python3 train.py --image_size 208 240 --num_classes 1 --folder_directory ../Data_Preparation/Axial/fold_3 --train_mri train_data_mri_fold_3.npy --train_mask train_data_mask_fold_3.npy --validation_mri validation_data_mri_fold_3.npy --validation_mask validation_data_mask_fold_3.npy --loss_function DiceCoefLoss --metrics_names precision recall  hausdorff_distance --epochs 150 --patients 20 --checkpoint_path ./ --checkpoint_name lo

# python3 train.py --image_size 208 240 --num_classes 1 --folder_directory ../Data_Preparation/Axial/fold_3 --train_mri train_data_mri_fold_3.npy --train_mask train_data_mask_fold_3.npy --validation_mri validation_data_mri_fold_3.npy --validation_mask validation_data_mask_fold_3.npy --loss_function TverskyLoss --metrics_names precision recall  hausdorff_distance --epochs 150 --patients 20 --checkpoint_path ./ --checkpoint_name lo

# python3 train.py --image_size 208 240 --num_classes 1 --folder_directory ../Data_Preparation/Axial/fold_3 --train_mri train_data_mri_fold_3.npy --train_mask train_data_mask_fold_3.npy --validation_mri validation_data_mri_fold_3.npy --validation_mask validation_data_mask_fold_3.npy --loss_function FocalLoss --metrics_names precision recall  hausdorff_distance --epochs 150 --patients 20 --checkpoint_path ./ --checkpoint_name lo

# python3 train.py --image_size 208 240 --num_classes 1 --folder_directory ../Data_Preparation/Axial/fold_3 --train_mri train_data_mri_fold_3.npy --train_mask train_data_mask_fold_3.npy --validation_mri validation_data_mri_fold_3.npy --validation_mask validation_data_mask_fold_3.npy --loss_function FocalTverskyLoss --metrics_names precision recall  hausdorff_distance --epochs 150 --patients 20 --checkpoint_path ./ --checkpoint_name lo
