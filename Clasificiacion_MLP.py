
# coding: utf-8

# In[1]:


# numerical computation
import numpy as np
import itertools
from random import randint
# import matplotlib and allow it to plot inline
import matplotlib.pyplot as plt

# import sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

# visualization library
import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set_context(rc={"font.family":'sans',"font.size":24,"axes.titlesize":24,"axes.labelsize":24})   

# dataframe management
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



# In[2]:


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


# In[3]:


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


# In[4]:


def maximun (df, name):
    maximun = df.sort_values(by='accuracy_validation',ascending=False).head(n=1)
    best = list(maximun[name])[0]
    return best


# In[5]:


def parameters(i):
    max_layers = 200
    min_layers = 10
    switcher={
            'activation':[ 'logistic', 'tanh', 'relu'],
            'solver':['sgd', 'adam'],
            'hidden_layer_sizes':[(i,) for i in range(min_layers,max_layers)]
         }
    return switcher.get(i,"Invalid parameters")


# In[6]:


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

# In[7]:


def hyper_MLP(path, features, name, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')
    
    param_grid = [
        {
            'activation':[ 'logistic', 'tanh', 'relu'],
            'solver':['sgd', 'adam'],
            'hidden_layer_sizes':[(i,) for i in range(min_layers,max_layers)]
        }
       ]
    MLP_evaluate=[]
    MLP_acc_model=[]
    MLP_std=[]

    best_solver=[]
    best_act=[]
    best_layer=[]

    model = MLPClassifier(max_iter=400, learning_rate_init=0.2,
                          learning_rate='invscaling', alpha = 1.0)
        
    for j in range(0, 50):
        print('Particion: ', j)
    #Normalizamos los datos de test y de train
        ss=StandardScaler()
        ss.fit(x_train[j][features])
        ss_train=ss.transform(x_train[j][features])
        #Buscamos los mejores parametros para esa división normalizada    
        clf = GridSearchCV(model, param_grid,
                           cv=KFold(n_splits=5), scoring='accuracy', n_jobs=-1)
        if multiclass==True:
            y_training = y_train[j].values.ravel()
        else:
            y_training = y_train[j]

        clf.fit(ss_train,y_training)

        best_index_Acc = np.nonzero(clf.cv_results_['rank_test_score'] == 1)[0][0]
        best_act.append(clf.best_params_['activation'])
        best_solver.append(clf.best_params_['solver'])
        best_layer.append(clf.best_params_['hidden_layer_sizes'])
        MLP_acc_model.append(clf.cv_results_['mean_test_score'][best_index_Acc])
        MLP_std.append(clf.cv_results_['std_test_score'][best_index_Acc])

        MLP_evaluate.append([best_act[j], best_solver[j], best_layer[j], round(MLP_acc_model[j],3),round(MLP_std[j],3)])

    labels_comp = ['activation', 'solver', 'hidden_layer_sizes', 'accuracy_validation', 'std']

    comparacion=pd.DataFrame(data=MLP_evaluate, columns = labels_comp)
    comparacion.to_csv('results/MLP/MLP_hyper_{}.csv'.format(name), index=False)


# In[8]:


def predict_MLP(path, features, name, multiclass=False):
    
    x_train, x_test, y_train, y_test = opened(path=path)
    print('Terminada la apertura de BBDD')

    comparacion.append(pd.read_csv('results/MLP/MLP_hyper_{}.csv'.format(name)))
    
    Activation = (maximun(comparacion, 'activation'))
    print(Activation)
    Solver = (maximun(comparacion, 'solver'))
    print(Solver)
    Layer = Layersint(maximun(comparacion, 'hidden_layer_sizes'))
    print(Layer)
    print('Con los parámetros óptimos procedemos a clasificar.')
    
    acuracy_ave=[]
    average_precision=[]
    average_recall=[]
    f1_scores=[]
    
    for i in range(0,50):
        ss=StandardScaler()
        ss.fit(x_train[i][features])
        ss_train=ss.transform(x_train[i][features])
        ss_test=ss.transform(x_test[i][features])

        clf= MLPClassifier(max_iter=400, learning_rate_init=0.2, activation=Activation, 
                           solver=Solver, hidden_layer_sizes= (Layer,), alpha = 1.0, 
                           learning_rate='invscaling')
        
        if multiclass==True:
            y_training = y_train[i].values.ravel()
        else:
            y_training = y_train[i]
               
        clf.fit(ss_train,y_training)

    #Predecimos el algoritmo con el mejor K
        y_true, y_pred = y_test[i], clf.predict(ss_test)

        average_precision.append(precision_score(y_true,y_pred, average='macro'))
        f1_scores.append(f1_score(y_true, y_pred, average='macro'))
        average_recall.append(recall_score(y_true, y_pred, average='macro'))
        acuracy_ave.append(accuracy_score(y_true, y_pred))
        
    predict=pd.DataFrame()
    predict['accuracy']=acuracy_ave
    predict['precision']=average_precision
    predict['recall']=average_recall
    predict['f1']=f1_scores
    predict.to_csv('results/MLP/MLP_predict_{}.csv'.format(name), index=False)


# In[9]:


boolean_class=[False, True, False, True]
import os.path as path


# In[10]:


names = ['all_all', 'all_ill'] 
features_freq = []
for n in names:
    with open("feature_selection/freq_{}.txt".format(n), "r") as file:
        features_freq.append(eval(file.readline()))


# In[11]:


paths_healthy = ['/label/O_WL_A_', '/class/O_WC_A_', '/label/P_WL_A_', '/class/P_WC_A_']
names_healthy=['freq_all_label_O', 'freq_all_class_O', 'freq_all_label_P', 'freq_all_class_P']
freq_features_healthy = features_freq[0] 


# In[12]:


for p, n, b in zip(paths_healthy, names_healthy, boolean_class):
    if path.exists('results/MLP/MLP_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_MLP(p, freq_features_healthy, n, b)
        print()
        print('--------------------------------------------------------')
        print()

    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, freq_features_healthy, n, b)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


paths_illness = ['/label/O_WL_WO_', '/class/O_WC_WO_', '/label/P_WL_WO_', '/class/P_WC_WO_']
names_illness=['freq_ill_label_O', 'freq_ill_class_O', 'freq_ill_label_P', 'freq_ill_class_P']
freq_features_illness = features_freq[1] 


# In[ ]:


for p, n, b in zip(paths_illness, names_illness, boolean_class):
    if path.exists('results/MLP/MLP_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_MLP(p, freq_features_illness, n, b)
        print()
        print('--------------------------------------------------------')
        print()

    if path.exists('results/MLP/MLP_predict_{}.csv'.format(n)): 
        print('Ya existe los resultados:', n)
    else:
        predict_MLP(p, freq_features_illness, n, b)
        print()
        print('--------------------------------------------------------')
        print()


# In[ ]:


names=['label_o_all_all','label_o_all_ill', 'label_p_all_all', 'label_p_all_ill']
features_rf_label = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_rf_label.append(eval(file.readline()))


# In[ ]:


path_label= ['/label/O_WL_A_', '/label/O_WL_WO_', '/label/P_WL_A_', '/label/P_WL_WO_']
names_label=['rf_all_label_O','rf_ill_label_O', 'rf_all_label_P', 'rf_ill_label_P']


# In[ ]:


for p, n, f in zip(path_label, names_label, features_rf_label):
    if path.exists('results/MLP/MLP_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_MLP(p, f, n)
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


names=['class_o_all_all','class_o_all_ill', 'class_p_all_all', 'class_p_all_ill']
features_rf_class = []
for n in names:
    with open("feature_selection/rf_{}.txt".format(n), "r") as file:
        features_rf_class.append(eval(file.readline()))


# In[ ]:


path_class= ['/class/O_WC_A_', '/class/O_WC_WO_', '/class/P_WC_A_', '/class/P_WC_WO_']
name_class=['rf_all_class_O','rf_ill_class_O', 'rf_all_class_P', 'rf_ill_class_P']


# In[ ]:


for p, n, f in zip(path_class, name_class, features_rf_class): 

    if path.exists('results/MLP/MLP_hyper_{}.csv'.format(n)): 
        print('Ya existe el hyperparametro:', n)
    else:
        hyper_MLP(p, f, n,True)
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


names = name_class + names_label + names_healthy + names_illness

param= ['activation', 'solver', 'hidden_layer_sizes']


# In[ ]:


for n in  names:
    print(n)
    print()
    hyper=[]
    for p in param:
        hyper.append(pd.read_csv('results/MLP/MLP_hyper_{}_{}.csv'.format(p, n)))
    Activation = (maximun(hyper[0], param[0]))
    print('Activation: ', Activation)
    Solver = (maximun(hyper[1], param[1]))
    print('Solver: ', Solver)
    Layers = Layersint(maximun(hyper[2], param[2]))
    print('Layers: ', Layers)
    
    predict = pd.read_csv('results/MLP/MLP_predict_{}.csv'.format(n))
    
    print('Tasa de acierto:', round(np.mean(predict['accuracy']), 4), '+/-', round(np.std(predict['accuracy']), 4))
    print('Tasa de precision', round(np.mean(predict['precision']), 4), '+/-', round(np.std(predict['precision']), 4))
    print('Tasa de exactitud:', round(np.mean(predict['recall']), 4),  '+/-', round(np.std(predict['recall']), 4))
    print('Tasa F1-Score', round(np.mean(predict['f1']), 4) , '+/-', round(np.std(predict['f1']),4))
    print('---------------------------------------------------------------')


# ## Escalado:
#     rf_all_class_O
# 
#     Activation:  relu
#     Solver:  adam
#     Layers:  112
#     Tasa de acierto: 0.7804 +/- 0.0198
#     Tasa de precision 0.7789 +/- 0.0235
#     Tasa de exactitud: 0.7668 +/- 0.0202
#     Tasa F1-Score 0.7683 +/- 0.0203
#     ---------------------------------------------------------------
#     rf_ill_class_O
# 
#     Activation:  relu
#     Solver:  adam
#     Layers:  21
#     Tasa de acierto: 0.8006 +/- 0.0182
#     Tasa de precision 0.7867 +/- 0.0189
#     Tasa de exactitud: 0.7723 +/- 0.0219
#     Tasa F1-Score 0.7756 +/- 0.0207
#     ---------------------------------------------------------------
#     rf_all_class_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  113
#     Tasa de acierto: 0.8503 +/- 0.0145
#     Tasa de precision 0.8439 +/- 0.0159
#     Tasa de exactitud: 0.8403 +/- 0.0147
#     Tasa F1-Score 0.8413 +/- 0.015
#     ---------------------------------------------------------------
#     rf_ill_class_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  145
#     Tasa de acierto: 0.8595 +/- 0.0174
#     Tasa de precision 0.8476 +/- 0.0194
#     Tasa de exactitud: 0.8358 +/- 0.0209
#     Tasa F1-Score 0.8408 +/- 0.02
#     ---------------------------------------------------------------
#     rf_all_label_O
# 
#     Activation:  logistic
#     Solver:  sgd
#     Layers:  107
#     Tasa de acierto: 0.742 +/- 0.0178
#     Tasa de precision 0.8945 +/- 0.0145
#     Tasa de exactitud: 0.8172 +/- 0.0175
#     Tasa F1-Score 0.8517 +/- 0.0119
#     ---------------------------------------------------------------
#     rf_ill_label_O
# 
#     Activation:  logistic
#     Solver:  sgd
#     Layers:  126
#     Tasa de acierto: 0.6964 +/- 0.0227
#     Tasa de precision 0.8762 +/- 0.0166
#     Tasa de exactitud: 0.8606 +/- 0.0147
#     Tasa F1-Score 0.8626 +/- 0.0114
#     ---------------------------------------------------------------
#     rf_all_label_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  146
#     Tasa de acierto: 0.862 +/- 0.0146
#     Tasa de precision 0.9312 +/- 0.0146
#     Tasa de exactitud: 0.8989 +/- 0.0157
#     Tasa F1-Score 0.9141 +/- 0.011
#     ---------------------------------------------------------------
#     rf_ill_label_P
# 
#     Activation:  relu
#     Solver:  adam
#     Layers:  24
#     Tasa de acierto: 0.8709 +/- 0.0169
#     Tasa de precision 0.9479 +/- 0.013
#     Tasa de exactitud: 0.9167 +/- 0.016
#     Tasa F1-Score 0.9308 +/- 0.0107
#     ---------------------------------------------------------------
#     freq_all_label_O
# 
#     Activation:  logistic
#     Solver:  sgd
#     Layers:  92
#     Tasa de acierto: 0.7079 +/- 0.0173
#     Tasa de precision 0.8746 +/- 0.0161
#     Tasa de exactitud: 0.7866 +/- 0.0167
#     Tasa F1-Score 0.8246 +/- 0.0119
#     ---------------------------------------------------------------
#     freq_all_class_O
# 
#     Activation:  logistic
#     Solver:  sgd
#     Layers:  83
#     Tasa de acierto: 0.7127 +/- 0.0194
#     Tasa de precision 0.7127 +/- 0.021
#     Tasa de exactitud: 0.6844 +/- 0.0198
#     Tasa F1-Score 0.6923 +/- 0.02
#     ---------------------------------------------------------------
#     freq_all_label_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  93
#     Tasa de acierto: 0.815 +/- 0.0159
#     Tasa de precision 0.8982 +/- 0.0149
#     Tasa de exactitud: 0.8508 +/- 0.0184
#     Tasa F1-Score 0.8723 +/- 0.0115
#     ---------------------------------------------------------------
#     freq_all_class_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  53
#     Tasa de acierto: 0.8161 +/- 0.0172
#     Tasa de precision 0.8071 +/- 0.019
#     Tasa de exactitud: 0.8018 +/- 0.0187
#     Tasa F1-Score 0.8032 +/- 0.0185
#     ---------------------------------------------------------------
#     freq_ill_label_O
# 
#     Activation:  logistic
#     Solver:  sgd
#     Layers:  135
#     Tasa de acierto: 0.6548 +/- 0.0245
#     Tasa de precision 0.8547 +/- 0.0169
#     Tasa de exactitud: 0.8295 +/- 0.0149
#     Tasa F1-Score 0.8332 +/- 0.0131
#     ---------------------------------------------------------------
#     freq_ill_class_O
# 
#     Activation:  relu
#     Solver:  adam
#     Layers:  145
#     Tasa de acierto: 0.7463 +/- 0.0187
#     Tasa de precision 0.7171 +/- 0.02
#     Tasa de exactitud: 0.7002 +/- 0.0237
#     Tasa F1-Score 0.7045 +/- 0.0223
#     ---------------------------------------------------------------
#     freq_ill_label_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  148
#     Tasa de acierto: 0.7943 +/- 0.0172
#     Tasa de precision 0.9054 +/- 0.0142
#     Tasa de exactitud: 0.8499 +/- 0.013
#     Tasa F1-Score 0.875 +/- 0.0107
#     ---------------------------------------------------------------
#     freq_ill_class_P
# 
#     Activation:  relu
#     Solver:  sgd
#     Layers:  41
#     Tasa de acierto: 0.7967 +/- 0.0156
#     Tasa de precision 0.7696 +/- 0.019
#     Tasa de exactitud: 0.7594 +/- 0.0199
#     Tasa F1-Score 0.7635 +/- 0.0193
#     ---------------------------------------------------------------
