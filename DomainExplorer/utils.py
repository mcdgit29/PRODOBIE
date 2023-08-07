def get_acc(y_true, y_pred):
    if all((y_true==1, y_pred==1)):
        return 'TP'
    elif all((y_true == 1, y_pred == 0)):
        return 'FN'
    elif all((y_true == 0, y_pred == 1)):
        return 'FP'
    elif all((y_true == 0, y_pred == 0)):
        return 'TN'
    else:
        raise ValueError(F'unknown input y_true: {y_true}  y_pred: {y_pred}')