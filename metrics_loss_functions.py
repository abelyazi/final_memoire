from keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn import metrics
import keras

class Avg(tf.keras.metrics.Metric):
    def __init__(self, name="avg", **kwargs):
        super().__init__(name=name, **kwargs)
        self.custom_auc = tf.keras.metrics.AUC(name='avg_auc',curve='PR')
        self.accuracy = tf.keras.metrics.Accuracy(name='avg_acc')
        self.fbeta_score = tf.keras.metrics.FBetaScore(beta=2.0, 
                                                       average='weighted', 
                                                       threshold=0.5,
                                                       name='avg_fbeta')
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.custom_auc.update_state(y_true, y_pred, sample_weight)
        self.accuracy.update_state(y_true, tf.round(y_pred), sample_weight)
        self.fbeta_score.update_state(y_true, tf.round(y_pred), sample_weight)
        self.total.assign(self.custom_auc.result() + self.accuracy.result() + self.fbeta_score.result())

    def result(self):
        return self.total / 3.0

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.custom_auc.reset_states()
        self.accuracy.reset_states()
        self.fbeta_score.reset_states()
        self.total.assign(0.0)
        
class Avg2(tf.keras.metrics.Metric):
    def __init__(self, name="avg2", **kwargs):
        super().__init__(name=name, **kwargs)
        self.custom_auc = tf.keras.metrics.AUC(name='avg_auc',curve='PR')
        self.fbeta_score = tf.keras.metrics.FBetaScore(beta=2.0, 
                                                       average='weighted', 
                                                       threshold=0.5,
                                                       name='avg_fbeta')
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.custom_auc.update_state(y_true, y_pred, sample_weight)
        self.fbeta_score.update_state(y_true, tf.round(y_pred), sample_weight)
        self.total.assign(self.custom_auc.result()  + self.fbeta_score.result())

    def result(self):
        return self.total / 2.0

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.custom_auc.reset_state()
        self.fbeta_score.reset_state()
        self.total.assign(0.0)
        
class CustomValidationLoss(tf.keras.callbacks.Callback):
    def __init__(self,xval,yval,dfval):
        self.xval = xval
        self.yval = yval
        self.dfval = dfval
    def on_epoch_end(self, epoch, logs=None):
        # Faire des prédictions sur les données de validation
        ypred = self.model.predict(self.xval)
        
        y_pred,y_val = get_ypred_ytrue(ypred,self.yval,self.dfval)
        # Calculer la perte de validation personnalisée
        custom_val_loss = loss_function(y_val, y_pred)  # Remplacer par votre fonction de perte personnalisée
        
        # Ajouter la perte de validation personnalisée aux journaux
        logs['custom_val_loss'] = custom_val_loss.numpy()
        
class AverageMetric(tf.keras.metrics.Metric):
    def __init__(self, name="AverageMetric", **kwargs):
        super().__init__(name=name, **kwargs)
        self.custom_auc = myAUC(name='my_auc')
        self.accuracy = tf.keras.metrics.Accuracy(name='acc')
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.custom_auc.update_state(y_true, y_pred, sample_weight)
        self.accuracy.update_state(y_true, tf.round(y_pred), sample_weight)
        self.total.assign(self.custom_auc.result() + self.accuracy.result())

    def result(self):
        return self.total / 2.0

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.custom_auc.reset_states()
        self.accuracy.reset_states()
        self.total.assign(0.0)

        
class myAUC(tf.keras.metrics.Metric):

    def __init__(self, name="custom_auc_pr", **kwargs):
        super(myAUC, self).__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(curve='PR')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return self.auc.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.auc.result()

    def reset_state(self):
        self.auc.reset_state()
       
    
def get_ypred_ytrue(ypred,yval,dfval):
    list_patient_rows = list(dfval.groupby(['patient_id'])['patient_id'].count())
    pred_groupby_patient = np.split(ypred,cumulative(list_patient_rows))
    pred_groupby_patient = pred_groupby_patient[:-1]
    dfval = dfval.astype({'murmur':'int'})
    y_true = dfval[['patient_id','murmur']].drop_duplicates()['murmur'].to_list()
    y_pred = majority_pred(pred_groupby_patient)
    return y_pred,np.array(y_true)


def weighted_binary_crossentropy(y_true, y_pred):
    pos_weight = class_weights[1]  # Set this to the value you want
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=pos_weight)
    return tf.reduce_mean(loss)


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras

def specificity(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def my_metric(y_true,y_pred):
    rec = recall(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    return class_weights[1]*rec + class_weights[0]*spec

def fbeta2(y_true, y_pred, beta=2):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)

def fbeta05(y_true, y_pred, beta=0.5):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r + K.epsilon())
    return K.mean(num / den)

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def negative_predictive_value(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))
    return tn / (tn + fn + K.epsilon())

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def f2_score(y_true,y_pred):
    y_pred = K.round(y_pred)
    return fbeta_score(y_true, y_pred, average='weighted', beta=2)

def fbeta_score(y_true, y_pred, beta=2):
    y_pred = K.round(y_pred)
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    fbeta_score = (1 + beta**2) * (p*r) / ((beta**2 * p) + r + K.epsilon())
    fbeta_score = tf.where(tf.math.is_nan(fbeta_score), tf.zeros_like(fbeta_score), fbeta_score)
    
    weights = tf.reduce_sum(y_true, axis=0)
    weights = weights / K.sum(weights)

    return K.sum(fbeta_score * weights)

def my_acc(y_true, y_pred):
    y_pred = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(y_true, y_pred), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct_predictions)
    return accuracy


def my_auc(y_true, y_pred):
    return metrics.roc_auc_score(y_true, y_pred)

def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.
    
    The weights are calculted by determining the number of 'pos' and 'neg' classes 
    in the true labels, then dividing by the number of total predictions.
    
    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.
    
    This can be useful for unbalanced catagories.

    """
    # get the total number of inputs
    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)
    
    # get weight of values in 'pos' category
    zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 
    
    # get weight of values in 'false' category
    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

    # calculate the weight vector
    weights =  (1.0 - true) * zero_weight +  true * one_weight 
    
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)

def weighted_bincrossentropy(true, pred, weight_zero = 0.6139, weight_one = 2.693):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.

    """
  
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)
    
    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return keras.backend.mean(weighted_bin_crossentropy)

def weighted_catcrossentropy(true, pred, weights=None):
    """
    Calculates weighted categorical cross entropy. The weights can be adjusted depending on class importance.
    
    This can be useful for unbalanced categories.
    
    Adjust the weights depending on what is required.
    The weights should be given in the order [weight_class_0, weight_class_1, weight_class_2]
    
    For example, if there are many more samples for class_0 than class_1 and class_2,
    you might use weights = [0.1, 1, 1]

    Parameters:
        true: actual labels
        pred: predicted labels
        weights: list of weights, one for each class.
    """
  
    if weights is None:
        weights = [3,5,1]
     
    # Ensure true labels are float32
    true = K.cast(true, 'float32')
    
    # clip to prevent NaN's and Inf's
    pred = K.clip(pred, K.epsilon(), 1-K.epsilon())
    
    # calculate weighted cross entropy
    cross_entropy = true * K.log(pred)
    weight_vector = true * weights
    weighted_cross_entropy = -weight_vector * cross_entropy

    return K.mean(weighted_cross_entropy, axis=-1)
  
