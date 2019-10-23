#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numerical computation
import numpy as np
import itertools
from random import randint

# import sklearn

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.model_selection import GridSearchCV

# dataframe management
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from IPython.display import clear_output


# In[ ]:


def opened (path=''):
    
    X_training=[]
    X_testing=[]
    y_training=[]
    y_testing=[]
           
    for j in range(0, 50):
        X_training.append(pd.read_csv('test_train_dataset{}{}_X_train.csv'.format(path,j)))
        X_testing.append(pd.read_csv('test_train_dataset{}{}_X_test.csv'.format(path, j)))
        y_training.append(pd.read_csv('test_train_dataset{}{}_y_train.csv'.format(path, j)))
        y_testing.append(pd.read_csv('test_train_dataset{}{}_y_test.csv'.format(path, j)))
        
    return X_training, X_testing, y_training, y_testing


# In[ ]:


def frequency (valor):
    max = 0
    res = list(valor)[0] 
    for i in list(valor): 
        freq = list(valor).count(i) 
        if freq > max: 
            max = freq 
            res = i 
    valor = res
    return valor


# In[ ]:


def maximun (df, name):
    maximun = df.sort_values(by='accuracy_model',ascending=False).head(n=1)
    best = list(maximun[name])[0]
    return best


# #  Aplicación del algoritmo RandomForest

# In[ ]:


def hyper_RF(path, features, name, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    hyper= pd.read_csv('results/DT/DT_hyper_{}.csv'.format(name))
    Depth = (maximun(hyper, 'max_depth'))
    Split = (maximun(hyper, 'min_samples_split'))
    
    turn_n_estimators = turn_n_estimators = [i for i in range(2,204)]
            
    param_grid = [
            {
                'n_estimators': turn_n_estimators
            }
           ]

    RF_evaluate=[]
    RF_acc_model=[]
    RF_error=[]
    RF_std=[]


    mean=[]
    std=[]
    best_estimator=[]
    
    
    
    for j in range(0, 50):
        droping=pd.concat([x_train[j][features], y_train[j]], axis=1,sort=False)
        droping=droping.drop_duplicates(subset=features, keep=False)
        xtrain= droping[features]
        if multiclass==True:
            ytrain=droping['CRG']
        else:
            ytrain=droping[['HP', 'Diabetes', 'Otros']]
                
        print('Particion: ', j)

    #Normalizamos x_test y x_train con la misma media y variancia que x_train
        ss=StandardScaler()
        ss.fit(xtrain)
        ss_train=ss.transform(xtrain)

    #Buscamos los mejores parametros para esa división normalizada
        clf = GridSearchCV(RandomForestClassifier(criterion='entropy',max_depth=Depth,
                                                  min_samples_split= Split), param_grid, 
                           scoring='accuracy',cv=KFold(n_splits=5), n_jobs=-1)
        
        if multiclass==True:
            y_training = ytrain.values.ravel()
        else:
            y_training = ytrain
        
        clf.fit(ss_train,y_training)

    #Evaluamos el algortimo teniendo en cuenta que para la función GridSearchCV test es nuestro train
        best_index_Acc = np.nonzero(clf.cv_results_['rank_test_score'] == 1)[0][0]
        
        best_estimator.append(clf.best_params_['n_estimators'])

        RF_acc_model.append(clf.cv_results_['mean_test_score'][best_index_Acc])
        RF_std.append(clf.cv_results_['std_test_score'][best_index_Acc])

        RF_evaluate.append([best_estimator[j], round(RF_acc_model[j],3), round(RF_std[j],3)])
    
             
    labels_comp = ['n_estimators', 'accuracy_model', 'std']

    comparacion=pd.DataFrame(data=RF_evaluate, columns = labels_comp)

    comparacion.to_csv('results/RF/RF_hyper_{}.csv'.format(name), index=False)

    Estimators = (maximun(comparacion, 'n_estimators'))
    print('Estimators: ', Estimators)


# In[ ]:


def predict_RF(path, features, name, multiclass=False):

    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    hyper= pd.read_csv('results/DT/DT_hyper_{}.csv'.format(name))
    Depth = (maximun(hyper, 'max_depth'))
    Split = (maximun(hyper, 'min_samples_split'))
    
    hyper= pd.read_csv('results/RF/RF_hyper_{}.csv'.format(name))
    Estimators = (maximun(hyper, 'n_estimators'))
    print('Estimators: ', Estimators)
    
    accuracy=[]
    hamming_losse=[]
    precision_macro=[]
    precision_micro=[]
    recall_macro=[]
    recall_micro=[]
    f1_scores_macro=[]
    f1_scores_micro=[]
    
    average_accuracy=[]
    average_precision=[]
    average_recall=[]
    f1_scores=[]
    
    for i in range(0,50):
        droping_train=pd.concat([x_train[i][features], y_train[i]], axis=1,sort=False)
        droping_train=droping_train.drop_duplicates(subset=features, keep=False)
        xtrain= droping_train[features]
        if multiclass==True:
            ytrain=droping_train['CRG']
        else:
            ytrain=droping_train[['HP', 'Diabetes', 'Otros']]

        droping_test=pd.concat([x_test[i][features], y_test[i]], axis=1,sort=False)
        droping=droping_test.drop_duplicates(subset=features, keep=False)
        xtest= droping_test[features]
        if multiclass==True:
            ytest=droping_test['CRG']
        else:
            ytest=droping_test[['HP', 'Diabetes', 'Otros']]
                
        ss=StandardScaler()
        ss.fit(xtrain)
        ss_train=ss.transform(xtrain)
        ss_test=ss.transform(xtest)

        clf= RandomForestClassifier(criterion='entropy',max_depth=Depth, 
                                    min_samples_split= Split, 
                                    n_estimators=Estimators)
        
        if multiclass==True:
            y_training = ytrain.values.ravel()
        else:
            y_training = ytrain
               
        clf.fit(ss_train,y_training)
        
        y_true, y_pred = ytest, clf.predict(ss_test)
        
        if multiclass==False:
            accuracy.append(accuracy_score(y_true, y_pred))
            hamming_losse.append(hamming_loss(y_true, y_pred))
            precision_macro.append(precision_score(y_true,y_pred, average='macro'))
            precision_micro.append(precision_score(y_true,y_pred, average='micro'))
            recall_macro.append(recall_score(y_true, y_pred, average='macro'))
            recall_micro.append(recall_score(y_true, y_pred, average='micro'))
            f1_scores_macro.append(f1_score(y_true, y_pred, average='macro'))
            f1_scores_micro.append(f1_score(y_true, y_pred, average='micro'))
        else:
            cm = confusion_matrix(y_true,y_pred)
            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            #num_classes = len(TP)
            #TN = []
            #for i in range(num_classes):
            #    temp = np.delete(cm, i, 0)    # delete ith row
            #    temp = np.delete(temp, i, 1)  # delete ith column
            #    TN.append(sum(sum(temp)))

            precision=TP / (TP + FP)
            recall=TP / (TP + FN)

            average_precision.append(np.mean(TP / (TP + FP)))
            average_recall.append(np.mean(TP / (TP + FN)))
            f1_scores.append(np.mean(2*(precision*recall)/(precision+recall)))
            average_accuracy.append(accuracy_score(y_true, y_pred))
        
    predict=pd.DataFrame()
    if multiclass==False:
        predict['accuracy']=accuracy
        predict['hamming_loss'] = hamming_losse
        predict['precision_macro']=precision_macro
        predict['precision_micro']=precision_micro
        predict['recall_macro']=recall_macro
        predict['recall_micro']=recall_micro
        predict['f1_macro']=f1_scores_macro
        predict['f1_micro']=f1_scores_micro
    else:
        predict['accuracy']=average_accuracy
        predict['precision']=average_precision
        predict['recall']=average_recall
        predict['f1']=f1_scores
        
    predict.to_csv('results/RF/RF_predict_{}.csv'.format(name), index=False)


# ## Bucles para las diferentes ejecuciones

# In[ ]:


import os.path as path


# ### Selección de Caracteriticas: Frecuencia

# In[ ]:


names = ['ocurrencia_all', 'ocurrencia_ill', 'presencia_all', 'presencia_ill'] 
features_freq = []
for n in names:
    with open("feature_selection/freq_{}.txt".format(n), "r") as file:
        features_freq.append(eval(file.readline()))


# In[ ]:


paths_CLASS = ['/class/O_WC_A_','/class/O_WC_WO_' , '/class/P_WC_A_', '/class/P_WC_WO_']
names_CLASS_fr=['freq_all_class_O', 'freq_ill_class_O', 'freq_all_class_P', 'freq_ill_class_P']


# In[ ]:


for p, n, f in zip(paths_CLASS, names_CLASS_fr, features_freq):
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


paths_LABEL = ['/label/O_WL_A_','/label/O_WL_WO_' , '/label/P_WL_A_', '/label/P_WL_WO_']
names_LABEL_fr=['freq_all_label_O', 'freq_ill_label_O', 'freq_all_label_P', 'freq_ill_label_P']


# In[ ]:


for p, n, f in zip(paths_LABEL, names_LABEL_fr, features_freq):
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()


# ### Selección de Caracteriticas: Random Forest

# In[ ]:


names=['label_o_all','label_o_ill', 'label_p_all', 'label_p_ill']
features_rf_label = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_rf_label.append(eval(file.readline()))


# In[ ]:


path_label= ['/label/O_WL_A_', '/label/O_WL_WO_', '/label/P_WL_A_', '/label/P_WL_WO_']
names_label_rf=['rf_all_label_O','rf_ill_label_O', 'rf_all_label_P', 'rf_ill_label_P']


# In[ ]:


for p, n, f in zip(path_label, names_label_rf, features_rf_label):
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


names=['class_o_all','class_o_ill', 'class_p_all', 'class_p_ill']
features_rf_class = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_rf_class.append(eval(file.readline()))


# In[ ]:


path_class= ['/class/O_WC_A_', '/class/O_WC_WO_', '/class/P_WC_A_', '/class/P_WC_WO_']
name_class_rf=['rf_all_class_O','rf_ill_class_O', 'rf_all_class_P', 'rf_ill_class_P']


# In[ ]:


for p, n, f in zip(path_class, name_class_rf, features_rf_class):  
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()


# ### Seleccion de frecuencia: F Fisher

# In[ ]:


names=['label_o_all','label_o_ill', 'label_p_all', 'label_p_ill']
features_fc_label = []
for n in names:
    with open("feature_selection/fc_{}.txt".format(n), "r") as file:
        features_fc_label.append(eval(file.readline()))


# In[ ]:


path_label= ['/label/O_WL_A_', '/label/O_WL_WO_', '/label/P_WL_A_', '/label/P_WL_WO_']
names_label_fc=['fc_all_label_O','fc_ill_label_O', 'fc_all_label_P', 'fc_ill_label_P']


# In[ ]:


for p, n, f in zip(path_label, names_label_fc, features_fc_label):
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


names=['class_o_all','class_o_ill', 'class_p_all', 'class_p_ill']
features_rf_class = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_rf_class.append(eval(file.readline()))


# In[ ]:


path_class= ['/class/O_WC_A_', '/class/O_WC_WO_', '/class/P_WC_A_', '/class/P_WC_WO_']
name_class_fc=['fc_all_class_O','fc_ill_class_O', 'fc_all_class_P', 'fc_ill_class_P']


# In[ ]:


for p, n, f in zip(path_class, name_class_fc, features_rf_class):
    if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_RF(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_RF(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()


# ## Resultados

# In[ ]:


def resultados_etiqueta(names):
    hyper_label=[]
    predict_label=[]
    for name in names:
        hyper_label.append(pd.read_csv('results/RF/RF_hyper_{}.csv'.format(name)))
        predict_label.append(pd.read_csv('results/RF/RF_predict_{}.csv'.format(name)))
    for i, n in zip(range(0, len(names)), names):
        print(n)
        print()
        Estimators = (maximun(hyper_label[i], 'n_estimators'))
        print('Estimators: ', Estimators)

        print('Tasa de acierto:', round(np.mean(predict_label[i]['accuracy']), 3), '+/-', round(np.std(predict_label[i]['accuracy']), 3))
        print('Tasa de Hamming Loss:', round(np.mean(predict_label[i]['hamming_loss']), 3), '+/-', round(np.std(predict_label[i]['hamming_loss']), 3))
        print('Tasa de precision(macro)', round(np.mean(predict_label[i]['precision_macro']), 3), '+/-', round(np.std(predict_label[i]['precision_macro']), 3))
        print('Tasa de precision(micro)', round(np.mean(predict_label[i]['precision_micro']), 3), '+/-', round(np.std(predict_label[i]['precision_micro']), 3))
        print('Tasa de exactitud(macro):', round(np.mean(predict_label[i]['recall_macro']), 3),  '+/-', round(np.std(predict_label[i]['recall_macro']), 3))
        print('Tasa de exactitud(micro):', round(np.mean(predict_label[i]['recall_micro']), 3),  '+/-', round(np.std(predict_label[i]['recall_micro']), 3))
        print('Tasa F1-Score(macro)', round(np.mean(predict_label[i]['f1_macro']), 3) , '+/-', round(np.std(predict_label[i]['f1_macro']),3))
        print('Tasa F1-Score(micro)', round(np.mean(predict_label[i]['f1_micro']), 3) , '+/-', round(np.std(predict_label[i]['f1_micro']),3))
        print('---------------------------------------------------------------')


# In[ ]:


resultados_etiqueta(names_LABEL_fr)


# In[ ]:


resultados_etiqueta(names_label_fc)


# In[ ]:


resultados_etiqueta(names_label_rf)


# In[ ]:


def resultados_clase(names):
    hyper_class=[]
    predict_class=[]
    for name in names:
        hyper_class.append(pd.read_csv('results/RF/RF_hyper_{}.csv'.format(name)))
        predict_class.append(pd.read_csv('results/RF/RF_predict_{}.csv'.format(name)))

    for i, n in zip(range(0, len(names)), names):
        print(n)
        print()
        Estimators = (maximun(hyper_class[i], 'n_estimators'))
        print('Estimators: ', Estimators)

        print('Tasa de acierto:', round(np.mean(predict_class[i]['accuracy']), 3), '+/-', round(np.std(predict_class[i]['accuracy']), 3))
        print('Tasa de precision', round(np.mean(predict_class[i]['precision']), 3), '+/-', round(np.std(predict_class[i]['precision']), 3))
        print('Tasa de exactitud:', round(np.mean(predict_class[i]['recall']), 3),  '+/-', round(np.std(predict_class[i]['recall']), 3))
        print('Tasa F1-Score', round(np.mean(predict_class[i]['f1']), 3) , '+/-', round(np.std(predict_class[i]['f1']),3))
        print('---------------------------------------------------------------')


# In[ ]:


resultados_clase(names_CLASS_fr)


# In[ ]:


resultados_clase(name_class_fc)


# In[ ]:


resultados_clase(name_class_rf)


# # Mejor resultados

# ## Multi-clase

# In[ ]:


names = ['fc_class_p_all', 'fc_class_p_ill']
names_features=['atc', 'cie', 'cie_atc']
features = []
for n in names:
    for f in names_features:
        with open("feature_selection/best/{}_{}.txt".format(n, f), "r") as file:
            features.append(eval(file.readline()))
    features += [['Edad', 'Sexo']]
paths_CLASS = ['/class/P_WC_A_', '/class/P_WC_WO_']
names_CLASS=['fc_all_class_P_atc', 'fc_all_class_P_cie', 'fc_all_class_P_cie_atc', 'fc_all_class_P_E_S', 
             'fc_ill_class_P_atc', 'fc_ill_class_P_cie', 'fc_ill_class_P_cie_atc', 'fc_ill_class_P_E_S']


# In[ ]:


k=['all','ill']
for p, i in zip(paths_CLASS, k):
    if i=='all':
        names = names_CLASS[0:4]
        feat = features[0:4]
    else:
        names = names_CLASS[4:8]
        feat = features[4:8]
    for n, f in zip( names, feat):
        if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
            print('Ya existe el hyperparametro:', n)
        else:
            hyper_RF(p, f, n, True)
            print()
            print('--------------------------------------------------------')
            print()

        if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
            print('Ya existe los resultados:', n)
        else:
            predict_RF(p, f, n, True)
            print()
            print('--------------------------------------------------------')
            print()


# In[ ]:


resultados_clase(names_CLASS)


# ## multi-label

# In[ ]:


names = ['fc_label_p_all', 'fc_label_p_ill']
names_features=['atc', 'cie', 'cie_atc']
features = []
for n in names:
    for f in names_features:
        with open("feature_selection/best/{}_{}.txt".format(n, f), "r") as file:
            features.append(eval(file.readline()))
    features += [['Edad', 'Sexo']]
paths_label = ['/label/P_WL_A_', '/label/P_WL_WO_']
names_label=['fc_all_label_P_atc', 'fc_all_label_P_cie', 'fc_all_label_P_cie_atc', 'fc_all_label_P_E_S', 
             'fc_ill_label_P_atc', 'fc_ill_label_P_cie', 'fc_ill_label_P_cie_atc', 'fc_ill_label_P_E_S']


# In[ ]:


k=['all','ill']
for p, i in zip(paths_label, k):
    if i=='all':
        names = names_label[0:4]
        feat = features[0:4]
    else:
        names = names_label[4:8]
        feat = features[4:8]
    for n, f in zip( names, feat):
        if path.exists('results/RF/RF_hyper_{}.csv'.format(n)): 
            print('Ya existe el hyperparametro:', n)
        else:
            hyper_RF(p, f, n)
            print()
            print('--------------------------------------------------------')
            print()

        if path.exists('results/RF/RF_predict_{}.csv'.format(n)): 
            print('Ya existe los resultados:', n)
        else:
            predict_RF(p, f, n)
            print()
            print('--------------------------------------------------------')
            print()


# In[ ]:


resultados_etiqueta(names_label)

