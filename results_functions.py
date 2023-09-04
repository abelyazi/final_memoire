import numpy as np
from sklearn.metrics import confusion_matrix


def check_max_pred(prd):
  max=0
  for i in prd:
    if i[0] > max:
      max = i[0]
  print(max)

def cumulative(lists): 
    cu_list = [] 
    length = len(lists) 
    cu_list = [sum(lists[0:x:1]) for x in range(0, length+1)] 
    return cu_list[1:]

def mean_pred(prediction):
  pred = []
  for i in prediction:
    lst = [ j[0] for j in i]
    avg = sum(lst)/len(lst)
    if avg > 0.5:
      pred.append(1)
    else:
      pred.append(0)
  return pred

def mean_pred_3_class(predictions):
    """
    Make predictions for 3-class classification based on the highest average probability and return one-hot encoded predictions.
    
    predictions : list of list of list
                  Each innermost list contains probabilities for the 3 classes.
    
    return : list of one-hot encoded predicted class labels
    """
    pred = []
    for group in predictions:
        avg_probs = [sum(p[i] for p in group)/len(group) for i in range(3)]
        pred_class = avg_probs.index(max(avg_probs))
        
        # One-hot encoding
        one_hot = [0] * 3
        one_hot[pred_class] = 1
        
        pred.append(one_hot)
        
    return pred

def majority_pred(prediction):
  pred = []
  for i in prediction:
    lst = [ j[0] for j in i]
    if (sum(elem > 0.5 for elem in lst)) > (len(lst)*0.5):
      pred.append(1)
    else:
      pred.append(0)
  return pred

def majority_pred_3_class(predictions):
    """
    Make predictions for 3-class classification based on majority rule and return one-hot encoded predictions.
    
    predictions : list of list of list
                  Each innermost list contains probabilities for the 3 classes.
    
    return : list of one-hot encoded predicted class labels
    """
    pred = []
    for group in predictions:
        votes = [sum(p[i] > 0.5 for p in group) for i in range(3)]
        pred_class = votes.index(max(votes))
        
        # One-hot encoding
        one_hot = [0] * 3
        one_hot[pred_class] = 1
        
        pred.append(one_hot)
        
    return pred

def get_mean_maj_preds_3_class(model_preds,df_test):
    list_patient_rows = list(df_test.groupby(['patient_id'])['patient_id'].count())
    pred_groupby_patient = np.split(model_preds,cumulative(list_patient_rows))
    pred_groupby_patient = pred_groupby_patient[:-1]
    #y_test_grouped = get_y_true3(df_test)
    mean_p = mean_pred_3_class(pred_groupby_patient)
    maj_p = majority_pred_3_class(pred_groupby_patient)
    return mean_p, maj_p

def get_mean_maj_preds(model_preds,df_test):
    list_patient_rows = list(df_test.groupby(['patient_id'])['patient_id'].count())
    pred_groupby_patient = np.split(model_preds,cumulative(list_patient_rows))
    pred_groupby_patient = pred_groupby_patient[:-1]
    y_test_grouped = get_y_true(df_test)
    mean_p = mean_pred(pred_groupby_patient)
    maj_p = majority_pred(pred_groupby_patient)
    return mean_p, maj_p

def compute_weight_acc(preds,labels):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    weight_positive = 5
    weight_negative = 1
    weighted_accuracy = (weight_positive * tp + weight_negative * tn) / (weight_positive * (tp + fn) + weight_negative * (tn + fp))
    return weighted_accuracy

def weighted_accuracy(y_true, y_pred, weights=[5,3,1]):
    # Ensure the true and predicted labels are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Convert one-hot encoded format to class label format
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Check for valid inputs
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    assert len(weights) == y_true.shape[1], "Number of weights must match the number of classes"

    # Compute instance weights based on true labels
    instance_weights = np.take(weights, y_true_labels)

    # Compute weighted accuracy
    correct_predictions = (y_true_labels == y_pred_labels).astype(int)
    return np.sum(correct_predictions * instance_weights) / np.sum(instance_weights)


def one_hot_to_label(one_hot_array):
    return np.argmax(one_hot_array, axis=1)

def get_y_true(df_test):
    df_test = df_test.astype({'murmur':'int'})
    y_test_grouped = df_test[['patient_id','murmur']].drop_duplicates()['murmur'].to_list()
    return y_test_grouped
import ast

def get_ytrue3(dftest):
    dftest['murmur'] = dftest['murmur'].apply(tuple)
    locdf = dftest[["patient_id","murmur"]]
    locdf = locdf.drop_duplicates()
    y_test_grouped = locdf['murmur'].to_list()
    y_test_grouped = [list(i) for i in y_test_grouped]
    return y_test_grouped