from sklearn.metrics import *
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon, entropy
from scipy.special import rel_entr
import numpy as np

def get_regression_metrics():
    d = {'ev': explained_variance_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
        'mape' : mean_absolute_percentage_error,
        'mean_predicted': lambda x, y: np.mean(y),
        'mean_actual': lambda x, y: np.mean(x),
        'qr_predicted_lower_q': lambda x, y: np.percentile(y, [25])[0],
        'qr_predicted_upper_q': lambda x, y: np.percentile(y, [75])[0],
        'qr_actual_lower_q': lambda x, y: np.percentile(x, [25])[0],
        'qr_actual_upper_q': lambda x, y: np.percentile(x, [75])[0],
        'label_entropy': lambda x,y: entropy(x),
        'prediction_entropy': lambda x, y: entropy(y)
        }
    return d

def get_binary_classification_metrics():
    d = {'roc_auc': roc_auc_score,
         'f1': f1_score,
         "balanced_accuracy_score": balanced_accuracy_score,
         'cohen_kappa': cohen_kappa_score,
         'accuracy_score':accuracy_score,
         'recall_score': recall_score,
         'precision_score': precision_score}
    return d
