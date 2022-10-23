from sklearn import model_selection as ms
from sklearn import preprocessing as pp
from sklearn import metrics
import pandas as pd 
import numpy as np
from imblearn.combine           import SMOTETomek
import scikitplot     as skplt
seed = 42


alvo='limite_adicional'

def encodes_obrigatorios(df, teste=True):
    num_cols = df.select_dtypes('number').drop('id_cliente', axis=1).columns


    map_bool = {'Sim': 1, 'Não': 0}
    map_alvo = {'Conceder': 1, 'Negar': 0}
    string_cols = ['investe_exterior', 'pessoa_polit_exp']

    df['investe_exterior'] = df['investe_exterior'].map(map_bool)
    df['pessoa_polit_exp'] = df['pessoa_polit_exp'].map(map_bool)
    return df


def cross_validation(model_name, model, x_train, y_train, threshold=0.5, ensemble=False):
    
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
        if ensemble == False:
            proba = model.predict_proba(x_test_cv)

            pred = (proba[:,1]>threshold).astype(int)
        else:
            pred = model.predict(x_test_cv)
        
        precision_list.append(metrics.precision_score(y_test_cv, pred))
        recall_list.append(metrics.recall_score(y_test_cv, pred))
        f1_list.append(metrics.f1_score(y_test_cv, pred, average='micro'))
        i+=1
        
    print(f'F1 : {np.mean(f1_list)} +/- {np.std(f1_list)}/nPrecision : {np.mean(precision_list)} +/- {np.std(precision_list)}\nRecall : {np.mean(recall_list)} +/- {np.std(recall_list)}\n')
    
def load_data():

    path = '../data/train.csv'

    df_raw = pd.read_csv(path)
    df_raw=encodes_obrigatorios(df_raw)
    
    map_alvo = {'Conceder': 1, 'Negar': 0}
    df_raw['limite_adicional'] = df_raw['limite_adicional'].map(map_alvo)

    df_test = pd.read_csv('../data/test.csv')
    df_test= encodes_obrigatorios(df_test)
    
    return df_raw, df_test


    
def single_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(metrics.f1_score(y_test, pred, average='micro'))


def outlier_treatment(df):
    df.loc[df['idade']>100, 'idade'] = df['idade'].median()

    # renda anual - baseado no percentil
    limit = np.percentile(df['renda_anual'], 0.95)
    df.loc[df['idade']>100, 'idade'] 

    df.loc[df['taxa_juros']>100, 'taxa_juros'] = df['taxa_juros'].median()
    
    return df

def feature_engineering(df):
    df['negativado'] = np.where(df['divida_atual']<df['saldo_atual'], 0, 1)
    df['divida/saldos_atuais']=df['divida_atual']/(df['saldo_atual']+df['valor_em_investimentos'])
    df['divida_atual/renda_mensal']=df['divida_atual']/(df['renda_anual']/12)
    df['saldo_liquido']=df['saldo_atual']+df['valor_em_investimentos']-df['divida_atual']
    df['score_credito']=df['saldo_liquido']*df['taxa_utilizacao_credito']
    df=df.drop(['divida_atual','saldo_atual',"valor_em_investimentos","saldo_liquido"], axis=1)
    return df

def data_preparation_train(df):
    mms = pp.MinMaxScaler()
    num_cols = df.select_dtypes('number').drop(['id_cliente','limite_adicional'], axis=1).columns
    df[num_cols]=mms.fit_transform(df[num_cols])
    return df

def data_preparation_test(df):
    mms = pp.MinMaxScaler()
  
    num_cols = df.select_dtypes('number').drop('id_cliente', axis=1).columns
    df[num_cols]=mms.fit_transform(df[num_cols])
    return df

def generate_submission(df_test,cols_selected, model):

    X_submission = df_test[cols_selected]
    pred = model.predict(X_submission)


    submission = pd.DataFrame()
    submission.loc[:, 'id_cliente'] = df_test['id_cliente']
    submission.loc[:, 'limite_adicional'] = pred
    submission.to_csv('../data/submission.csv', index=False)
    print('Congratulations! Submission created!')
    
   
def balance_features(X_train_imb, y_train_imb):
    
    synthetic_samples = SMOTETomek(sampling_strategy = 'minority', random_state = seed, n_jobs = 14)

    X_train, y_train = synthetic_samples.fit_resample(X_train_imb, y_train_imb)
    return X_train, y_train
    
def plot_cm(model, X_test, y_test):

    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, pred);





        
        