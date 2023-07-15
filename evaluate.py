from sklearn.metrics import precision_score, recall_score, f1_score, \
    accuracy_score, matthews_corrcoef, cohen_kappa_score, \
    brier_score_loss, auc,roc_curve, confusion_matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt

def spe(Y_test, Y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    specificity = tn/(tn+fp)
    return specificity


def model_evaluation(train_true, train_pred, train_T_prob):
    precision = precision_score(train_true, train_pred)
    acc = accuracy_score(train_true, train_pred)
    mcc = matthews_corrcoef(train_true, train_pred)
    f1 = f1_score(train_true, train_pred)
    kappa = cohen_kappa_score(train_true, train_pred)
    BS = brier_score_loss(train_true,train_T_prob)
    FPR, TPR, threshold = roc_curve(train_true,train_T_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    specificity = spe(train_true, train_pred)
    # Sensitivity=recall
    sensitivity = recall_score(train_true, train_pred)
    BAC = (sensitivity+specificity)/2

    return [AUC, acc, sensitivity, specificity, BAC, f1, kappa, mcc, precision, BS]



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    cb = plt.colorbar(drawedges = False)
    cb.ax.tick_params(labelsize=25)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize = 25)
    plt.yticks(tick_marks, classes, fontsize = 25)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')
    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],fontsize=30,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label',fontsize=25)
    plt.xlabel('Predicted label',fontsize=25)
    plt.tight_layout()


def plot_AUC(true_train, train_T_score, tasks_name):
    for i in range(7):
        FPR,TPR,threshold = roc_curve(true_train[i] , train_T_score[i], pos_label=1)
        plt.plot(FPR, TPR, lw=4, label='{} (Area = {})'.format(tasks_name[i],round(auc(FPR,TPR),3)), linestyle="-")
    for i in range(7,12):
        FPR,TPR,threshold = roc_curve(true_train[i] , train_T_score[i], pos_label=1)
        plt.plot(FPR, TPR, lw=4, label='{} (area = {})'.format(tasks_name[i],round(auc(FPR,TPR),3)),linestyle = "-.")
    plt.plot([0, 1], [0, 1], color='m', lw=7, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xticks(fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('False Positive Rate',fontsize = 30)
    plt.ylabel('True Positive Rate',fontsize = 30)
    plt.legend(loc="lower right", fontsize=30)