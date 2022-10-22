from sklearn import model_selection as ms
import pandas as pd 
import numpy as np
from sklearn import metrics


def cv(model_name, model, x_train, y_train, threshold=0.5):
    
    precision_list = []
    recall_list = []
    f1_list = []

    skf = ms.StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
    i =1
    for train_index, test_index in skf.split(x_train, y_train):
        print(f'{i}/5')
        x_train_cv = x_train.iloc[train_index]
        y_train_cv = y_train.iloc[train_index]
        
        x_test_cv = x_train.iloc[test_index]
        y_test_cv = y_train.iloc[test_index]
        
        # model training
        model.fit(x_train_cv, y_train_cv)
        
         # prediction
        proba = model.predict_proba(x_test_cv)
        
        pred = (proba[:,1]>threshold).astype(int)
        
        precision_list.append(metrics.precision_score(y_test_cv, pred))
        recall_list.append(metrics.recall_score(y_test_cv, pred))
        f1_list.append(metrics.f1_score(y_test_cv, pred, average='micro'))
        i+=1
        
    print(f'{model_name}\nF1 : {np.round(np.mean(f1_list),2)} +/- {np.round(np.std(f1_list),2)}\n Precision : {np.round(np.mean(precision_list),2)} +/- {np.round(np.std(precision_list),2)}\nRecall : {np.round(np.mean(recall_list),2)} +/- {np.round(np.std(recall_list),2)}\n')
    
