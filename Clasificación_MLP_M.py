#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# numerical computation
import numpy as np
import itertools
from random import randint


# import sklearn

from sklearn.neural_network import MLPClassifier
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
    maximun = df.sort_values(by='accuracy_validation',ascending=False).head(n=1)
    best = list(maximun[name])[0]
    return best


# In[ ]:


def parameters(i):
    max_layers = 60#204
    min_layers = 10#5
    switcher={
            'activation':[ 'tanh', 'relu'],
            'solver': ['sgd'], #['sgd', 'adam'],
            'hidden_layer_sizes':[(i,) for i in range(min_layers,max_layers, 2)]
         }
    return switcher.get(i,"Invalid parameters")


# In[ ]:


def Layersint(Layers):
    Layers = tuple(Layers)
    maxi=len(Layers)
    cal=0
    res=[]
    for i in range(0, maxi):
        if Layers[i]=="(" or Layers[i]=="," or Layers[i]==")":
            res += str(Layers[i])
        else:
            if maxi==6:
                if i==1:
                    cal = int(Layers[i])*100
                elif i==2:
                    cal += int(Layers[i])*10
                else:
                    cal +=int(Layers[i])
            elif maxi==5:
                if i==1:
                    cal = int(Layers[i])*10
                else:
                    cal += int(Layers[i])
            else:
                cal = int(Layers[i])

    return cal


# #  Aplicación del algoritmo MLP

# In[ ]:


def hyper_MLP(path, features, name, param, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    param_grid = [
        {
            param: parameters(param)
        }
       ]
    MLP_evaluate=[]
    MLP_acc_model=[]
    MLP_std=[]

    best_param=[]
    
    if param == 'hidden_layer_sizes':
        rest = ['activation', 'solver']
        resto=[]
        for r in rest:
            resto.append(pd.read_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(r, name)))

        Activation = (maximun(resto[0], rest[0]))
        print(Activation)
        Solver = (maximun(resto[1], rest[1]))
        print(Solver)
        
        model = MLPClassifier(max_iter=400, learning_rate_init=0.2, 
                              learning_rate='invscaling', alpha = 1.0, 
                              solver=Solver, activation=Activation) 
    else:
        model = MLPClassifier(max_iter=400, learning_rate_init=0.2,
                              learning_rate='invscaling', alpha = 1.0)
        
    for j in range(0, 50):
        
        droping=pd.concat([x_train[j][features], y_train[j]], axis=1,sort=False)
        droping=droping.drop_duplicates(subset=features, keep=False)
        xtrain= droping[features]
        if multiclass==True:
            ytrain=droping['CRG']
        else:
            ytrain=droping[['HP', 'Diabetes', 'Otros']]
                
        print('Particion: ', j)
    #Normalizamos los datos de test y de train
        ss=StandardScaler()
        ss.fit(xtrain)
        ss_train=ss.transform(xtrain)
        #Buscamos los mejores parametros para esa división normalizada    
        clf = GridSearchCV(model, param_grid,
                           cv=KFold(n_splits=5), scoring='accuracy', n_jobs=-1)
        if multiclass==True:
            y_training = ytrain.values.ravel()
        else:
            y_training = ytrain

        clf.fit(ss_train,y_training)

        best_index_Acc = np.nonzero(clf.cv_results_['rank_test_score'] == 1)[0][0]
        best_param.append(clf.best_params_[param])
        MLP_acc_model.append(clf.cv_results_['mean_test_score'][best_index_Acc])
        MLP_std.append(clf.cv_results_['std_test_score'][best_index_Acc])

        MLP_evaluate.append([best_param[j],  round(MLP_acc_model[j],3),round(MLP_std[j],3)])

    labels_comp = [param , 'accuracy_validation', 'std']

    comparacion=pd.DataFrame(data=MLP_evaluate, columns = labels_comp)
    comparacion.to_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(param, name), index=False)


# In[ ]:


def predict_MLP(path, features, name, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    param= ['activation', 'solver', 'hidden_layer_sizes']
    comparacion=[]
    for p in param:
        comparacion.append(pd.read_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(p, name)))
    
    Activation = (maximun(comparacion[0], param[0]))
    print(Activation)
    Solver = (maximun(comparacion[1], param[1]))
    print(Solver)
    Layer = Layersint(maximun(comparacion[2], param[2]))
    print(Layer)
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

        clf= MLPClassifier(max_iter=400, learning_rate_init=0.2, activation=Activation, 
                           solver=Solver, hidden_layer_sizes= (Layer,), alpha = 1.0, 
                           learning_rate='invscaling')
        
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
    predict.to_csv('results/MLP/MLP_predict_{}.csv'.format(name), index=False)


# ### Selección de Caracteriticas: Frecuencia

# In[ ]:


import os.path as path
param= ['activation', 'solver', 'hidden_layer_sizes']


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
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m, True)
            print()
            print('--------------------------------------------------------')
            print()

    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


paths_LABEL = ['/label/O_WL_A_','/label/O_WL_WO_' , '/label/P_WL_A_', '/label/P_WL_WO_']
names_LABEL_fr=['freq_all_label_O', 'freq_ill_label_O', 'freq_all_label_P', 'freq_ill_label_P']


# In[ ]:


for p, n, f in zip(paths_LABEL, names_LABEL_fr, features_freq):
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m)
            print()
            print('--------------------------------------------------------')
            print()
    
    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n)
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
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m)
            print()
            print('--------------------------------------------------------')
            print()
    
    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n)
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
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m,True)
            print()
            print('--------------------------------------------------------')
            print()
    
    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n, True)
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
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m)
            print()
            print('--------------------------------------------------------')
            print()
    
    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


names=['class_o_all','class_o_ill', 'class_p_all', 'class_p_ill']
features_fc_class = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_fc_class.append(eval(file.readline()))


# In[ ]:


path_class= ['/class/O_WC_A_', '/class/O_WC_WO_', '/class/P_WC_A_', '/class/P_WC_WO_']
name_class_fc=['fc_all_class_O','fc_ill_class_O', 'fc_all_class_P', 'fc_ill_class_P']


# In[ ]:


for p, n, f in zip(path_class, name_class_fc, features_fc_class):
    for m in param:        
        if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
            print('Ya existe el hyperparametro:', n, m)
        else:
            hyper_MLP(p, f, n, m,True)
            print()
            print('--------------------------------------------------------')
            print()
    
    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, f, n, True)
        print()
        print('--------------------------------------------------------')
        print()


# ### Resultados:

# In[ ]:


def resultados_etiquetas(names):
    for n in  names:
        print(n)
        print()
        hyper=[]
        for p in param:
            hyper.append(pd.read_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(p, n)))
        Activation = (maximun(hyper[0], param[0]))
        print(Activation)
        Solver = (maximun(hyper[1], param[1]))
        print(Solver)
        Layer = Layersint(maximun(hyper[2], param[2]))
        print(Layer)

        predict = pd.read_csv('results/MLP/MLP_predict_{}.csv'.format(n))

        print('Tasa de acierto:', round(np.mean(predict['accuracy']), 3), '+/-', round(np.std(predict['accuracy']), 3))
        print('Tasa de Hamming Loss:', round(np.mean(predict['hamming_loss']), 3), '+/-', round(np.std(predict['hamming_loss']), 3))
        print('Tasa de precision(macro)', round(np.mean(predict['precision_macro']), 3), '+/-', round(np.std(predict['precision_macro']), 3))
        print('Tasa de precision(micro)', round(np.mean(predict['precision_micro']), 3), '+/-', round(np.std(predict['precision_micro']), 3))
        print('Tasa de exactitud(macro):', round(np.mean(predict['recall_macro']), 3),  '+/-', round(np.std(predict['recall_macro']), 3))
        print('Tasa de exactitud(micro):', round(np.mean(predict['recall_micro']), 3),  '+/-', round(np.std(predict['recall_micro']), 3))
        print('Tasa F1-Score(macro)', round(np.mean(predict['f1_macro']), 3) , '+/-', round(np.std(predict['f1_macro']),3))
        print('Tasa F1-Score(micro)', round(np.mean(predict['f1_micro']), 3) , '+/-', round(np.std(predict['f1_micro']),3))
        print('---------------------------------------------------------------')


# In[ ]:


resultados_etiquetas(names_LABEL_fr)


# In[ ]:


resultados_etiquetas(names_label_fc)


# In[ ]:


resultados_etiquetas(names_label_rf)


# In[ ]:


def resultados_clases(names):
    for n in  names:
        print(n)
        print()
        hyper=[]
        for p in param:
            hyper.append(pd.read_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(p, n)))
        Activation = (maximun(hyper[0], param[0]))
        print(Activation)
        Solver = (maximun(hyper[1], param[1]))
        print(Solver)
        Layer = Layersint(maximun(hyper[2], param[2]))
        print(Layer)

        predict = pd.read_csv('results/MLP/MLP_predict_{}.csv'.format(n))

        print('Tasa de acierto:', round(np.mean(predict['accuracy']), 3), '+/-', round(np.std(predict['accuracy']), 3))
        print('Tasa de precision', round(np.mean(predict['precision']), 3), '+/-', round(np.std(predict['precision']), 3))
        print('Tasa de exactitud:', round(np.mean(predict['recall']), 3),  '+/-', round(np.std(predict['recall']), 3))
        print('Tasa F1-Score', round(np.mean(predict['f1']), 3) , '+/-', round(np.std(predict['f1']),3))
        print('---------------------------------------------------------------')


# In[ ]:


resultados_clases(names_CLASS_fr)


# In[ ]:


resultados_clases(name_class_fc)


# In[ ]:


resultados_clases(name_class_rf)


# # Mejor configuración

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
        for m in param:        
            if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
                print('Ya existe el hyperparametro:', n, m)
            else:
                hyper_MLP(p, f, n, m, True)
                print()
                print('--------------------------------------------------------')
                print()

        if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
            print('Ya existe los resultados:', n)
        else:
            predict_MLP(p, f, n, True)
            print()
            print('--------------------------------------------------------')
            print()


# In[ ]:


resultados_clases(names_CLASS)


# ## Multi-label

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
        for m in param:        
            if path.exists('results/MLP/MLP_hyper_{}_{}.csv'.format(m,n)): 
                print('Ya existe el hyperparametro:', n, m)
            else:
                hyper_MLP(p, f, n, m)
                print()
                print('--------------------------------------------------------')
                print()

        if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
            print('Ya existe los resultados:', n)
        else:
            predict_MLP(p, f, n)
            print()
            print('--------------------------------------------------------')
            print()


# In[ ]:


resultados_etiquetas(names_label)


# In[ ]:




