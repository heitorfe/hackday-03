import random
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest,\
VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


import hackday

def tuning_rf(X_train, X_test, y_train, y_test, K=10):
    results_rf = pd.DataFrame()
    count = 0

    for i in range(K):

        max_depth = random.choice(np.arange(70,100,20))
        min_samples_split = random.choice([2,3,4,5])
        min_child_weight = random.choice(np.arange(0.001,0.003,0.0005))
        min_samples_leaf = random.choice([1,2,3])
        n_estimators = random.choice(np.arange(200,1000,100))
    #     n_estimators = random.choice(np.arange(100,200,10))


        model = RandomForestClassifier(n_estimators = n_estimators,
                              max_depth =max_depth ,
                              min_samples_leaf=min_samples_leaf,
                              min_samples_split=min_samples_split,
                              class_weight={0:1 ,1:10})

        model.fit(X_train, y_train)
        threshold = hackday.optimal_threshold(model, X_test, y_test)
        f1 =  hackday.cross_validation('rf', model, X_train, y_train, threshold=threshold, ensemble=False)

        results_rf.loc[count, 'f1'] = f1
        results_rf.loc[count, 'threshold'] = threshold
        results_rf.loc[count, 'n_estimators'] = n_estimators
        results_rf.loc[count, 'max_depth'] = max_depth
        results_rf.loc[count, 'min_samples_leaf'] = min_samples_leaf
        results_rf.loc[count, 'min_samples_split'] = min_samples_split

        count+=1
        print(count)
        results_rf.sort_values('f1', ascending=False).to_pickle('./tunning/rf.pkl')
    return results_rf.sort_values('f1', ascending=False)
        
        
def tuning_lgbm(X_train, X_test, y_train, y_test, K=10):
    results = pd.DataFrame()
    count = 0
    for i in range(K):

        learning_rate = random.choice(np.arange(0.15, 0.2, 0.005))
        num_leaves = random.choice(np.arange(70,90,5))
        max_depth = random.choice(np.arange(50,150,20))
        min_child_weight = random.choice(np.arange(0.001,0.003,0.0005))
        n_estimators = random.choice(np.arange(50,500,50))
        boosting_type = 'gbdt'
    #     n_estimators = random.choice(np.arange(100,200,10))


        model = LGBMClassifier(learning_rate = learning_rate,
                              num_leaves = num_leaves,
                              max_depth =max_depth ,
                             n_estimators = n_estimators,
                              min_child_weight=min_child_weight,
                              boosting_type= boosting_type)

        model.fit(X_train, y_train)
        threshold = hackday.optimal_threshold(model, X_test, y_test)
        f1 =  hackday.cross_validation('lgbm', model, X_train, y_train, threshold=threshold, ensemble=False)

        results.loc[count, 'f1'] = f1
        results.loc[count, 'threshold'] = threshold
        results.loc[count, 'learning_rate'] = learning_rate
        results.loc[count, 'num_leaves'] = num_leaves
        results.loc[count, 'n_estimators'] = n_estimators
        results.loc[count, 'min_child_weight'] = min_child_weight
        results.loc[count, 'max_depth'] = max_depth
        count+=1
        print(count)
        results.sort_values('f1', ascending=False).to_pickle('./tunning/lgbm.pkl')
    return results.sort_values('f1', ascending=False)

def tuning_ada(X_train, X_test, y_train, y_test, K=10):
    results_ada = pd.DataFrame()
    count = 0

    for i in range(10):

        learning_rate = random.choice(np.arange(1.0,1.5,0.005))
        n_estimators = random.choice(np.arange(50,500,50))
        algorithm = random.choice(['SAMME', 'SAMME.R'])
    #     n_estimators = random.choice(np.arange(100,200,10))

        model = AdaBoostClassifier(base_estimator=None,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        algorithm=algorithm,
        random_state=42,)

        model.fit(X_train, y_train)
        threshold = hackday.optimal_threshold(model, X_test, y_test)
        f1 =  hackday.cross_validation('ada', model, X_train, y_train, threshold=threshold, ensemble=False)

        results_ada.loc[count, 'f1'] = f1
        results_ada.loc[count, 'threshold'] = threshold
        results_ada.loc[count, 'n_estimators'] = n_estimators
        results_ada.loc[count, 'learning_rate'] = learning_rate
        results_ada.loc[count, 'algorithm'] = algorithm

        count+=1
        print(count)
        results_ada.sort_values('f1', ascending=False).to_pickle('./tunning/ada.pkl')
    return results_ada.sort_values('f1', ascending=False)

def tuning_et(X_train, X_test, y_train, y_test, K=10):
    results_et = pd.DataFrame()
    count = 0

    for i in range(K):
        max_depth = random.choice(np.arange(50,150,20))
        criterion = random.choice(['gini', 'entropy'])
        min_samples_leaf = random.choice([1,2,3])
        min_samples_split = random.choice([2,3,4,5])
        n_estimators = random.choice(np.arange(100,1000,100))

        model = ExtraTreesClassifier(
        max_depth = max_depth, 
        n_estimators=n_estimators,
        criterion=criterion,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight={0:1 ,1:10})

        model.fit(X_train, y_train)
        threshold = hackday.optimal_threshold(model, X_test, y_test)
        f1 =  hackday.cross_validation('et', model, X_train, y_train, threshold=threshold, ensemble=False)

        results_et.loc[count, 'f1'] = f1
        results_et.loc[count, 'threshold'] = threshold
        results_et.loc[count, 'n_estimators'] = n_estimators
        results_et.loc[count, 'max_depth'] = max_depth
        results_et.loc[count, 'criterion'] = criterion
        results_et.loc[count, 'min_samples_leaf'] = min_samples_leaf
        results_et.loc[count, 'min_samples_split'] = min_samples_split

        count+=1
        print(count)
        results_et.sort_values('f1', ascending=False).to_pickle('./tunning/et.pkl')
    return results_et.sort_values('f1', ascending=False)


def tuning_xgb(X_train, X_test, y_train, y_test, K=10):
    results_xgb = pd.DataFrame()
    count = 0

    for i in range(K):
        n_estimators = random.choice(np.arange(100,1000,100))
        max_depth = random.choice(np.arange(3,10,1))
        subsample = random.choice(np.arange(0.7,0.9,0.02))
        colsample_bytree = random.choice(np.arange(0,1,0.01))
        gamma = random.choice([1,2,3])

        

        model = XGBClassifier(
        max_depth = max_depth, 
        n_estimators=n_estimators,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        random_state=42)

        model.fit(X_train, y_train)
        threshold = hackday.optimal_threshold(model, X_test, y_test)
        f1 =  hackday.cross_validation('xgb', model, X_train, y_train, threshold=threshold, ensemble=False)

        results_xgb.loc[count, 'f1'] = f1
        results_xgb.loc[count, 'threshold'] = threshold
        results_xgb.loc[count, 'n_estimators'] = n_estimators
        results_xgb.loc[count, 'max_depth'] = max_depth
        results_xgb.loc[count, 'subsample'] = subsample
        results_xgb.loc[count, 'colsample_bytree'] = colsample_bytree
        results_xgb.loc[count, 'gamma'] = gamma

        count+=1
        print(count)
        results_xgb.sort_values('f1', ascending=False).to_pickle('./tunning/xgb.pkl')
    return results_xgb.sort_values('f1', ascending=False)