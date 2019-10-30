#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numerical computation
import numpy as np
import itertools
from random import randint
# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt

# import sklearn

from sklearn.neighbors import KNeighborsClassifier

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
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)


# ## Funciones a utilizar

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


# #  Aplicación del algoritmo KNN Clasificador.
# 

# In[ ]:


def hyper_KNN(path, features, name, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    max_neighbors = 50
    turn_neighbors = [i for i in range(1,max_neighbors)]

    param_grid = [
            {
                'n_neighbors' : turn_neighbors
            }
           ]

    KNN_evaluate=[]
    KNN_acc_model=[]
    KNN_std=[]

    average = []

    best_k=[]
    print('Empezamos a buscar los méjores parámetros')
    for j in range(0, 50):
        
        droping=pd.concat([x_train[j][features], y_train[j]], axis=1,sort=False)
        droping=droping.drop_duplicates(subset=features, keep=False)
        xtrain= droping[features]
        if multiclass==True:
            ytrain=droping['CRG']
        else:
            ytrain=droping[['HP', 'Diabetes', 'Otros']]
        print('Partición: ', j)

    #Normalizamos los datos de test y de train
        ss=StandardScaler()
        ss.fit(xtrain)
        ss_train=ss.transform(xtrain)

    #Bus* camos el mejor K para esa división normalizada    
        clf = GridSearchCV(KNeighborsClassifier(n_jobs=-1),param_grid, 
                              scoring='accuracy', cv=KFold(n_splits=5), 
                              n_jobs=-1)
        if multiclass==True:
            y_training = ytrain.values.ravel()
        else:
            y_training = ytrain

        clf.fit(ss_train,y_training)

    #Evaluamos el algortimo teniendo en cuenta que para la función GridSearchCV test es nuestro train
        best_index_Acc = np.nonzero(clf.cv_results_['rank_test_score'] == 1)[0][0]
        best_k.append(clf.best_params_['n_neighbors'])
        KNN_acc_model.append(clf.cv_results_['mean_test_score'][best_index_Acc])
        KNN_std.append(clf.cv_results_['std_test_score'][best_index_Acc])

        average.append(clf.cv_results_['mean_test_score'])

        KNN_evaluate.append([best_k[j], round(KNN_acc_model[j],3), round(KNN_std[j],3)])#W
    
        
    to_plot= pd.DataFrame(data= average, columns=turn_neighbors)
    
    to_plot.to_csv('results/KNN/KNN_to_plot_{}.csv'.format(name), index=False)
        
    labels_comp = ['k','accuracy_model', 'std']

    comparacion=pd.DataFrame(data=KNN_evaluate, columns = labels_comp)

    comparacion.to_csv('results/KNN/KNN_hyper_{}.csv'.format(name), index=False)


# In[ ]:


def predict_KNN(path, features, name, multiclass=False): 
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    comparacion = pd.read_csv('results/KNN/KNN_hyper_{}.csv'.format(name))
    Neigh = maximun(comparacion, 'k')
    print('Neigh: ', Neigh)
    
    print('Con los parámetros óptimos procedemos a clasificar.')
    
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

        clf= KNeighborsClassifier(n_neighbors =Neigh, n_jobs=-1)
        if multiclass==True:
            y_training = ytrain.values.ravel()
        else:
            y_training = ytrain
        clf.fit(ss_train,y_training)

    #Predecimos el algoritmo con el mejor K
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
        
    predict.to_csv('results/KNN/KNN_predict_{}.csv'.format(name), index=False)


#     max_neighbors = 155
#     turn_neighbors = [i for i in range(55,max_neighbors, 2)]
#     df=to_plot.transpose()
#     plt.figure()
#     df.plot(legend=False)
#     plt.xticks()
#     plt.grid()
#     plt.show()

# ## Bucles para las diferentes ejecuciones

# In[ ]:


boolean_class=[False, True, False, True]
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
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()

    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


paths_LABEL = ['/label/O_WL_A_','/label/O_WL_WO_' , '/label/P_WL_A_', '/label/P_WL_WO_']
names_LABEL_fr=['freq_all_label_O', 'freq_ill_label_O', 'freq_all_label_P', 'freq_ill_label_P']


# In[ ]:


for p, n, f in zip(paths_LABEL, names_LABEL_fr, features_freq):
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()

    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n)
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
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n)
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
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()


# ### Selección de caracteristicas: F Fisher

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
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


names=['class_o_all','class_o_ill', 'class_p_all', 'class_p_ill']
features_fc_class = []
for n in names:
    with open("feature_selection/fc_{}.txt".format(n), "r") as file:
        features_fc_class.append(eval(file.readline()))


# In[ ]:


path_class= ['/class/O_WC_A_', '/class/O_WC_WO_', '/class/P_WC_A_', '/class/P_WC_WO_']
name_class_fc=['fc_all_class_O','fc_ill_class_O', 'fc_all_class_P', 'fc_ill_class_P']


# In[ ]:


for p, n, f in zip(path_class, name_class_fc, features_fc_class):
    if path.exists('results/KNN/KNN_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_KNN(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()
    
    if path.exists('results/KNN/KNN_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_KNN(p, f, n, multiclass=True)
        print()
        print('--------------------------------------------------------')
        print()


# --------------------------------------

# ## Resultados

# In[ ]:


def resultados_etiqueta(names):
    hyper_label=[]
    predict_label=[]
    for name in names:
        hyper_label.append(pd.read_csv('results/KNN/KNN_hyper_{}.csv'.format(name)))
        predict_label.append(pd.read_csv('results/KNN/KNN_predict_{}.csv'.format(name)))
    for i, n in zip(range(0, len(names)), names):
        print(n)
        print()
        Neigh = maximun(hyper_label[i], 'k')
        print('Neigh: ', Neigh)

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


def resultados_clases(names):
    hyper_class=[]
    predict_class=[]
    for name in names:
        hyper_class.append(pd.read_csv('results/KNN/KNN_hyper_{}.csv'.format(name)))
        predict_class.append(pd.read_csv('results/KNN/KNN_predict_{}.csv'.format(name)))
    for i, n in zip(range(0, len(names)), names):
        print(n)
        print()
        Neigh = maximun(hyper_class[i], 'k')
        print('Neigh: ', Neigh)

        print('Tasa de acierto:', round(np.mean(predict_class[i]['accuracy']), 3), '+/-', round(np.std(predict_class[i]['accuracy']), 3))
        print('Tasa de precision', round(np.mean(predict_class[i]['precision']), 3), '+/-', round(np.std(predict_class[i]['precision']), 3))
        print('Tasa de exactitud:', round(np.mean(predict_class[i]['recall']), 3),  '+/-', round(np.std(predict_class[i]['recall']), 3))
        print('Tasa F1-Score', round(np.mean(predict_class[i]['f1']), 3) , '+/-', round(np.std(predict_class[i]['f1']),3))
        print('---------------------------------------------------------------')


# In[ ]:


resultados_clases(names_CLASS_fr)


# In[ ]:


resultados_clases(name_class_fc)


# In[ ]:


resultados_clases(name_class_rf)


# In[ ]:





# In[ ]:




