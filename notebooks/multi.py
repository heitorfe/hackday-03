import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, IsolationForest,\
VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from multiprocessing import Process
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import hackday
import fine_tuning
alvo = 'limite_adicional'
cols_selected  = ['taxa_juros',
 'dias_atraso_dt_venc',
 'num_pgtos_atrasados',
 'num_cartoes_credito',
 'divida/saldos_atuais',
 'num_contas_bancarias',
 'idade',
 'divida_atual/renda_mensal',
 'num_consultas_credito',
 'taxa_utilizacao_credito',
 'score_credito',
 'renda_anual',
 'num_emprestimos']

if __name__ == "__main__": 
    # load data
    df_raw, df_test = hackday.load_data()

    # outlier treatment
    df = hackday.outlier_treatment(df_raw)

#     df = hackday.feature_engineering(df)

    df = hackday.data_preparation_train(df)

    X = df.drop(['id_cliente', 'limite_adicional'], axis=1)
    y = df['limite_adicional']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


    procs = []

    proc = Process(target=fine_tuning.tuning_rf, args=(X_train, X_test, y_train, y_test, 50,))
    procs.append(proc)
    proc.start()

    proc = Process(target=fine_tuning.tuning_ada, args=(X_train, X_test, y_train, y_test,50,))
    procs.append(proc)
    proc.start()

    proc = Process(target=fine_tuning.tuning_lgbm, args=(X_train, X_test, y_train, y_test,50,))
    procs.append(proc)
    proc.start()
    
    proc = Process(target=fine_tuning.tuning_et, args=(X_train, X_test, y_train, y_test,50,))
    procs.append(proc)
    proc.start()
    
    proc = Process(target=fine_tuning.tuning_xgb, args=(X_train, X_test, y_train, y_test,50,))
    procs.append(proc)
    proc.start()

    for proc in procs:
        proc.join()