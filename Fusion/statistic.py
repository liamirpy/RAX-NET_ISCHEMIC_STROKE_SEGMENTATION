import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np

import scipy.ndimage as ndi


class Statistics:
    smooth = 1



    @staticmethod
    def ADAMW(initial_learning_rate):
        """
        Define the optimizer for the model.

        This function defines the optimizer to be used during model training.

        Returns:
        op : AdamW optimizer
            AdamW optimizer with specified parameters.
        """
        # Define learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.9
        )
        
        # Define AdamW optimizer
        op = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.004,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        return op








    @staticmethod
    def FocalTverskyLoss(ground_truth, predictions):
        """
        Calculate the custom loss function.

        This function computes a custom loss function, which is a combination of various metrics such as
        intersection over union, Tversky index, and binary focal loss.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        loss : tensor
            Computed loss value.
        """
        ground_truth = K.cast(ground_truth, 'float32')

        predictions = K.cast(predictions, 'float32')

        floss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.80,
            gamma=2,
            from_logits=False,
            label_smoothing=0.0,
            axis=-1
        )
        focal_loss = floss(ground_truth, predictions)
                
        ground_truth_flat = K.flatten(ground_truth)
        
        predictions_flat = K.flatten(predictions)
        


        # tversky=(true_positives + Statistics.smooth) / (true_positives + 0.3 * false_positives + 0.7 * false_negatives + Statistics.smooth)



        true_positives = K.sum(ground_truth_flat * predictions_flat)


        false_positives = K.sum((predictions_flat - (ground_truth_flat * predictions_flat)))

        false_negatives = K.sum((ground_truth_flat - (ground_truth_flat * predictions_flat)))


        tversky=(true_positives + Statistics.smooth) / (true_positives + 0.3 * false_positives + 0.7 * false_negatives + Statistics.smooth)




        return 1 - tversky + focal_loss 
    


    @staticmethod
    def FocalLoss(ground_truth, predictions):
        """
        Calculate the focal loss function.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        loss : tensor
            Computed loss value.
        """
        ground_truth = K.cast(ground_truth, 'float32')

        predictions = K.cast(predictions, 'float32')

        floss = keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=0.80,
            gamma=2,
            from_logits=False,
            label_smoothing=0.0,
            axis=-1
        )
        focal_loss = floss(ground_truth, predictions)

   

        return focal_loss
    




    @staticmethod
    def TverskyLoss(ground_truth, predictions):
        """
        Calculate the Tversky loss function.


        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        loss : tensor
            Computed loss value.
        """
        ground_truth = K.cast(ground_truth, 'float32')

        predictions = K.cast(predictions, 'float32')

                
        ground_truth_flat = K.flatten(ground_truth)

        predictions_flat = K.flatten(predictions)
        
        true_positives = K.sum(ground_truth_flat * predictions_flat)


        false_positives = K.sum((predictions_flat - (ground_truth_flat * predictions_flat)))

        false_negatives = K.sum((ground_truth_flat - (ground_truth_flat * predictions_flat)))


        tversky=(true_positives + Statistics.smooth) / (true_positives + 0.3 * false_positives + 0.7 * false_negatives + Statistics.smooth)



        return 1 - tversky


        # y_true_pos = K.flatten(ground_truth)
        # y_pred_pos = K.flatten(predictions)
        # true_pos = K.sum(y_true_pos * y_pred_pos)
        # false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        # false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        # alpha = 0.7
        # return 1 -(true_pos + 1)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + 1)


    @staticmethod
    def DiceCoefLoss(ground_truth, predictions):
        """
        Calculate the Dice coefficient.

        This function computes the Dice coefficient, which measures the similarity between two samples.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        dice_coefficient : tensor
            Dice coefficient.
        """
        ground_truth = K.cast(ground_truth, 'float32')
        predictions = K.cast(predictions, 'float32')

        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)

        intersection = K.sum(ground_truth_flat * predictions_flat)

        return 1- ((2 * intersection + Statistics.smooth) / (K.sum(ground_truth_flat) + K.sum(predictions_flat) + Statistics.smooth))





    @staticmethod
    def hausdorff_distance(y_true, y_pred):
        """
        Compute the Hausdorff distance between two sets of points.

        This function calculates the Hausdorff distance, which measures the maximum distance of a set
        to the nearest point in another set.

        Parameters:
        y_true : tensor
            Ground truth values. Shape (batch_size, num_points, 2) for 2D points.
        y_pred : tensor
            Predicted values. Shape (batch_size, num_points, 2) for 2D points.
        
        Returns:
        loss : tensor
            Computed Hausdorff distance.
        """
        def compute_hausdorff(a, b):
            """
            Compute pairwise Hausdorff distance between two sets of points.

            Parameters:
            a : tensor
                Set of points.
            b : tensor
                Another set of points.
            
            Returns:
            hausdorff : tensor
                Pairwise Hausdorff distance.
            """
            # Compute pairwise distances
            d_matrix = tf.norm(tf.expand_dims(a, axis=-2) - tf.expand_dims(b, axis=-3), axis=-1)
            
            # Directed distances
            d_ab = tf.reduce_max(tf.reduce_min(d_matrix, axis=-1), axis=-1)
            d_ba = tf.reduce_max(tf.reduce_min(d_matrix, axis=-2), axis=-1)
            
            # Hausdorff distance
            hausdorff = tf.maximum(d_ab, d_ba)
            
            return hausdorff
        
        # Compute batch-wise Hausdorff distances
        hd_batch = tf.map_fn(lambda x: compute_hausdorff(x[0], x[1]), (y_true, y_pred), dtype=tf.float32)
        
        # Return the mean Hausdorff distance for the batch
        return tf.reduce_mean(hd_batch)



    @staticmethod
    def dice_coef(ground_truth, predictions):
        """
        Calculate the Dice coefficient.

        This function computes the Dice coefficient, which measures the similarity between two samples.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        dice_coefficient : tensor
            Dice coefficient.
        """
        ground_truth = K.cast(ground_truth, 'float32')
        predictions = K.cast(predictions, 'float32')

        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)
        predictions_rounded = K.round(predictions_flat)

        intersection = K.sum(ground_truth_flat * predictions_rounded)

        return ((2 * intersection + Statistics.smooth) / (K.sum(ground_truth_flat) + K.sum(predictions_rounded) + Statistics.smooth))
    




    @staticmethod
    def TP(ground_truth, predictions):
        """
        Calculate the number of true positives.

        This function computes the number of true positives, which are the correct positive predictions
        made by a model.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        true_positives : tensor
            Number of true positives.
        """
        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)
        true_positives = K.sum(K.round(K.clip(ground_truth_flat * predictions_flat, 0, 1)))
        return true_positives

    # Similar comments for FP, TN, FN, rec, pre, our_loss, and dice_coef methods...

    @staticmethod
    def FP(ground_truth, predictions):
        """
        Calculate the number of false positives.

        This function computes the number of false positives, which are the incorrect positive predictions
        made by a model.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        false_positives : tensor
            Number of false positives.
        """
        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)
        predictions_clipped = K.round(K.clip(predictions_flat, 0, 1))
        true_positives_clipped = K.round(K.clip(ground_truth_flat * predictions_flat, 0, 1))
        false_positives = K.sum(K.round(K.clip(predictions_clipped - true_positives_clipped, 0, 1)))
        return false_positives

    @staticmethod
    def TN(ground_truth, predictions):
        """
        Calculate the number of true negatives.

        This function computes the number of true negatives, which are the correct negative predictions
        made by a model.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        true_negatives : tensor
            Number of true negatives.
        """
        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)
        predictions_clipped = K.round(K.clip(predictions_flat, 0, 1))
        all_one = K.ones_like(predictions_clipped)
        predictions_complement = -1 * (predictions_clipped - all_one)
        ground_truth_complement = -1 * (ground_truth_flat - all_one)
        true_negatives = K.sum(K.round(K.clip(ground_truth_complement + predictions_complement, 0, 1)))
        return true_negatives

    @staticmethod
    def FN(ground_truth, predictions):
        """
        Calculate the number of false negatives.

        This function computes the number of false negatives, which are the incorrect negative predictions
        made by a model.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        false_negatives : tensor
            Number of false negatives.
        """
        ground_truth_flat = K.flatten(ground_truth)
        predictions_flat = K.flatten(predictions)
        true_positives_clipped = K.round(K.clip(ground_truth_flat * predictions_flat, 0, 1))
        false_negatives = K.sum(K.round(K.clip(ground_truth_flat - true_positives_clipped, 0, 1)))
        return false_negatives






    @staticmethod
    def recall(ground_truth, predictions):
        """
        Calculate the recall score.

        This function computes the recall score, which measures the ability of a model to find all the
        relevant cases within a dataset.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        recall : tensor
            Recall score.
        """
        tp = Statistics.TP(ground_truth, predictions)
        fn = Statistics.FN(ground_truth, predictions)
        return (tp + 1) / (tp + 1 + fn)

    @staticmethod
    def precision(ground_truth, predictions):
        """
        Calculate the precision score.

        This function computes the precision score, which measures the ability of a model to identify
        only the relevant data points.

        Parameters:
        ground_truth : tensor
            Ground truth values.
        predictions : tensor
            Predicted values.
        
        Returns:
        precision : tensor
            Precision score.
        """
        tp = Statistics.TP(ground_truth, predictions)
        fp = Statistics.FP(ground_truth, predictions)
        return (tp + 1) / (tp + 1 + fp)
    



    ### 3D evaluation



    @staticmethod
    def lesion_f1_score(truth, prediction, empty_value=1.0):
        tp, fp, fn = 0, 0, 0
        f1_score = empty_value

        labeled_ground_truth, num_lesions = ndi.label(truth.astype(bool))

        # For each true lesion, check if there is at least one overlapping voxel. This determines true positives and
        # false negatives (unpredicted lesions)
        for idx_lesion in range(1, num_lesions+1):
            lesion = labeled_ground_truth == idx_lesion
            lesion_pred_sum = lesion + prediction
            if(np.max(lesion_pred_sum) > 1):
                tp += 1
            else:
                fn += 1

        # For each predicted lesion, check if there is at least one overlapping voxel in the ground truth.
        labaled_prediction, num_pred_lesions = ndi.label(prediction.astype(bool))
        for idx_lesion in range(1, num_pred_lesions+1):
            lesion = labaled_prediction == idx_lesion
            lesion_pred_sum = lesion + truth
            if(np.max(lesion_pred_sum) <= 1):  # No overlap
                fp += 1

        # Compute f1_score
        denom = tp + (fp + fn)/2
        if(denom != 0):
            f1_score = tp / denom
        return f1_score




    @staticmethod
    def simple_lesion_count(ground_truth, predictions):
        _, num_lesions_gt = ndi.label(ground_truth)
        _, num_lesions_pr = ndi.label(predictions)
        
        return np.abs(num_lesions_gt - num_lesions_pr)




    @staticmethod
    def volume_difference(ground_truth, predictions):
        return np.abs(np.sum(ground_truth) - np.sum(predictions))
