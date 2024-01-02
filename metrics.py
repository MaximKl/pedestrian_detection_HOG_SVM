from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def get_metrics(name, y_true, y_pred):
    print(f"\n{name} prediction accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"{name} precision score: {precision_score(y_true, y_pred)}")
    print(f"{name} recall score: {recall_score(y_true, y_pred)}")
    print(f"{name} F1 score: {f1_score(y_true, y_pred)}")
    confusion = confusion_matrix(y_true, y_pred)
    print(f"{name} confusion matrix:\n{confusion}")