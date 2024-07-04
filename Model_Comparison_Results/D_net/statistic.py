import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras.losses import binary_crossentropy
import keras.backend as K

class Statistics:
    smooth = 1



    @staticmethod
    def EML(y_true, y_pred):
        # gamma = 1.1
        # alpha = 0.48
        # smooth = 1.
        # y_true = K.flatten(y_true)
        # y_pred = K.flatten(y_pred)
        # intersection = K.sum(y_true*y_pred)
        # dice_loss = (2.*intersection + smooth)/(K.sum(y_true*y_true)+K.sum(y_pred * y_pred)+smooth)
        # y_pred = K.clip(y_pred, K.epsilon(), None)
        # pt_1 = tf.where(tf.equal(y_true, 1),y_pred,tf.ones_like(y_pred))
        # pt_0 = tf.where(tf.equal(y_true, 0),y_pred,tf.zeros_like(y_pred))
        # focal_loss = -K.mean(alpha*K.pow(1. -pt_1, gamma)*K.log(pt_1),axis=-1)\
        #             -K.mean(1-alpha)*K.pow(pt_0,gamma)*K.log((1. -pt_0),axis=-1)
        return Statistics.FocalLoss(y_true, y_pred) - K.log(Statistics.DiceCoefLoss(y_true, y_pred))
    




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
            alpha=0.48,
            gamma=1.1,
            from_logits=False,
            label_smoothing=0.0,
            axis=-1
        )
        focal_loss = floss(ground_truth, predictions)

   

        return focal_loss
    



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
