import csv
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split, cross_val_score, KFold
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import SVC
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
#0:'age  1:'workclass'  2:'fnlwgt'    3:'education'     4:'education_num'
#5:'marital_status'   6:'occupation'  7:'relationship'  8:'race'
#9:'sex'    10:'capital_gain'   11:'capital_loss'   12:'hours_per_week'
#13:'native_country'
#Numerical  : 0, 2, 4, 10
#Categorial : 1, 3, 5, 6, 7 ,8, 9

# Correlation matrix between numerical values
#numeric_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#g = sns.heatmap(dataset[numeric_features].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
#plt.show()
#======Income 統計圖=======
#sns.countplot(dataset[14],label="Count")
#sns.plt.show()
#==============================================
#======不同feature x 跟income y 的關係==========
'''
g = sns.factorplot(x=1, y=14,data=dataset,kind="bar",size = 8,palette = "muted")
g.despine(left=True)
g = g.set_ylabels(">50K probability")
'''
def income_pred(train_file, test_file):
    dataset = pd.read_csv(train_file, header=None)
    dataset.isnull().sum()
    #========deal with features====================
    #workclass
    dataset.iloc[:,1].replace([' Without-pay'], 0, inplace = True)
    dataset.iloc[:,1].replace([' Private', ' State-gov', ' Self-emp-not-inc', ' Local-gov'], 
                                1, inplace = True)
    dataset.iloc[:,1].replace([' Federal-gov'], 2, inplace = True)
    dataset.iloc[:,1].replace([' Self-emp-inc'], 3, inplace = True)
    #education
    dataset.iloc[:,3].replace([' 12th', ' 9th', ' 11th', ' 7th-8th', ' 10th', ' 1st-4th', ' 5th-6th', ' Preschool'], 
                                0, inplace = True)
    dataset.iloc[:,3].replace([' HS-grad', ' Some-college', ' Assoc-acdm', ' Assoc-voc'], 
                                1, inplace = True)
    dataset.iloc[:,3].replace([' Masters', ' Bachelors', ' Prof-school', ' Doctorate'], 
                                2, inplace = True)
    #marital_status
    dataset.iloc[:,5].replace([' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], 
                                0, inplace = True)
    dataset.iloc[:,5].replace([' Married-AF-spouse', ' Married-civ-spouse'], 1, inplace = True)
    #occupation
    dataset.iloc[:,6].replace([' Other-service', ' Priv-house-serv', ' Handlers-cleaners'], 
                                0, inplace = True)
    dataset.iloc[:,6].replace([' Adm-clerical', ' Farming-fishing', ' Machine-op-inspct'], 
                                1, inplace = True)
    dataset.iloc[:,6].replace([' Tech-support', ' Craft-repair', ' Sales', ' Transport-moving'], 
                                2, inplace = True)
    dataset.iloc[:,6].replace([' Exec-managerial', ' Prof-specialty', ' Protective-serv'], 
                                3, inplace = True)
    dataset.iloc[:,6].replace([' Armed-Forces'], 4, inplace = True)
    #relationship
    dataset.iloc[:,7].replace([' Not-in-family', ' Own-child', ' Unmarried', ' Other-relative'], 
                                0, inplace = True)
    dataset.iloc[:,7].replace([' Husband',' Wife'], 1, inplace = True)
    #race
    dataset.iloc[:,8].replace([' Black', ' Amer-Indian-Eskimo', ' Other'], 0, inplace = True)
    dataset.iloc[:,8].replace([' White', ' Asian-Pac-Islander'], 1, inplace = True)
    #sex
    dataset.iloc[:,9].replace([' Female'], 0, inplace = True)
    dataset.iloc[:,9].replace([' Male'], 1, inplace = True)
    #native_country
    dataset.iloc[:,9].replace([' Female'], 0, inplace = True)
    dataset.iloc[:,9].replace([' Female'], 0, inplace = True)
    #==============================================
    #drop the features we don't use : 11:'capital_loss' and 13:'native_country
    dataset.drop(labels=[13], axis = 1, inplace = True)
    ##Random Forest
    array = dataset.values
    X = array[:, 0:13]
    Y = array[:, 13]
    validation_size = 0.20
    seed = 7
    num_folds = 10
    scoring = 'accuracy'
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
        test_size=validation_size, random_state=seed)
    #==============Random Forest==================
    #param_test = {'max_depth':list(range(1,16,1)), 'min_samples_split':list(range(100,801,100))}
    #gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, n_estimators=260, min_samples_leaf=20, 
    #      max_features='sqrt', subsample=0.8, random_state=10), 
    #       param_grid = param_test, scoring='roc_auc',iid=False, cv=5)
    #param_test1 = {'n_estimators':list(range(120,401,10))}
    #gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, min_samples_split=700,
    #                                  min_samples_leaf=20,max_depth=10,max_features='sqrt', subsample=0.8,random_state=10), 
    #                       param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
    #param_test = {'min_samples_split':list(range(500,2100,100)), 'min_samples_leaf':list(range(30,101,10))}
    #gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130,max_depth=10,
    #                                     max_features='sqrt', subsample=0.8, random_state=10), 
    #                       param_grid = param_test, scoring='roc_auc',iid=False, cv=5)
    #param_test4 = {'max_features':list(range(1,21,1))}
    #gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=130,max_depth=10, min_samples_leaf =30, 
    #              min_samples_split =1700, subsample=0.8, random_state=10), 
    #                       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    
    #
    '''
    param_test = {'max_features':list(range(2,13,1))}
    gsearch = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, n_estimators=260,max_depth=10, min_samples_leaf=30, 
                        min_samples_split =1600, subsample=0.9, random_state=10), 
                           param_grid = param_test, scoring='roc_auc',iid=False, cv=10)
    '''
    #gsearch.fit(X,Y)
    #print(gsearch.grid_scores_)
    #print(gsearch.best_params_)
    #print(gsearch.best_score_)
    
    
#    gbc = GradientBoostingClassifier(learning_rate=0.05, n_estimators=260, max_depth=10, min_samples_leaf=30, 
#                   min_samples_split =1700, max_features=11, subsample=0.9, random_state=10)
    gbc = GradientBoostingClassifier(learning_rate=0.05, n_estimators=260, max_depth=5, min_samples_leaf=30, 
                   min_samples_split =1700, max_features=11, subsample=0.9, random_state=10)   
    scores = cross_val_score(gbc, X, Y, cv=5, scoring='accuracy')
    train_model = gbc.fit(X,Y)
    
    #learning rate=0.1時 n_estimator:130, max_depth=10, min_samples_leaf=30, min_sampe_split=1700 
    #                   max_feautures = 11
    #===========find max_depth===================
    '''
    param_test2 = {'max_depth':list(range(3,20,2)), 'min_samples_split':list(range(50,201,20))}
    gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90, 
                                      min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
       param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
    gsearch2.fit(X,Y)
    max_depth': 19, 'min_samples_split': 50
    '''
    #===========find min_samples_leaf and min_samples_split===================
    '''
    param_test3 = {'min_samples_split':list(range(10,150,5)), 'min_samples_leaf':list(range(10,100,5))}
    gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90, max_depth=19,
                                      max_features='sqrt' ,oob_score=True, random_state=10),
       param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
    gsearch3.fit(X,Y)
    'min_samples_leaf': 10, 'min_samples_split': 40
    '''
    #===========find max_features===================
    '''
    param_test4 = {'max_features':list(range(2,14,1))}
    gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 90, max_depth=19, min_samples_split=40,
                                      min_samples_leaf=10 ,oob_score=True, random_state=10),
       param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
    gsearch4.fit(X,Y)
    print(gsearch4.grid_scores_)
    print(gsearch4.best_params_)
    print(gsearch4.best_score_)
    'max_features': 3
    '''
    #==============================================
    #rf2 = RandomForestClassifier(n_estimators= 90, max_depth=19, min_samples_split=40,
    #                                  min_samples_leaf=10,max_features=3 ,oob_score=True, random_state=10)
    #train_model = rf2.fit(X,Y)
    #print (rf2.oob_score_)
    #=======================Test data===============================
    test_data = pd.read_csv(test_file, header=None)
    
    #========deal with features====================
    #workclass
    test_data.iloc[:,1].replace([' Without-pay'], 0, inplace = True)
    test_data.iloc[:,1].replace([' Private', ' State-gov', ' Self-emp-not-inc', ' Local-gov'], 
                                1, inplace = True)
    test_data.iloc[:,1].replace([' Federal-gov'], 2, inplace = True)
    test_data.iloc[:,1].replace([' Self-emp-inc'], 3, inplace = True)
    #education
    test_data.iloc[:,3].replace([' 12th', ' 9th', ' 11th', ' 7th-8th', ' 10th', ' 1st-4th', ' 5th-6th', ' Preschool'], 
                                0, inplace = True)
    test_data.iloc[:,3].replace([' HS-grad', ' Some-college', ' Assoc-acdm', ' Assoc-voc'], 
                                1, inplace = True)
    test_data.iloc[:,3].replace([' Masters', ' Bachelors', ' Prof-school', ' Doctorate'], 
                                2, inplace = True)
    #marital_status
    test_data.iloc[:,5].replace([' Divorced', ' Married-spouse-absent', ' Never-married', ' Separated', ' Widowed'], 
                                0, inplace = True)
    test_data.iloc[:,5].replace([' Married-AF-spouse', ' Married-civ-spouse'], 1, inplace = True)
    #occupation
    test_data.iloc[:,6].replace([' Other-service', ' Priv-house-serv', ' Handlers-cleaners'], 
                                0, inplace = True)
    test_data.iloc[:,6].replace([' Adm-clerical', ' Farming-fishing', ' Machine-op-inspct'], 
                                1, inplace = True)
    test_data.iloc[:,6].replace([' Tech-support', ' Craft-repair', ' Sales', ' Transport-moving'], 
                                2, inplace = True)
    test_data.iloc[:,6].replace([' Exec-managerial', ' Prof-specialty', ' Protective-serv'], 
                                3, inplace = True)
    test_data.iloc[:,6].replace([' Armed-Forces'], 4, inplace = True)
    #relationship
    test_data.iloc[:,7].replace([' Not-in-family', ' Own-child', ' Unmarried', ' Other-relative'], 
                                0, inplace = True)
    test_data.iloc[:,7].replace([' Husband',' Wife'], 1, inplace = True)
    #race
    test_data.iloc[:,8].replace([' Black', ' Amer-Indian-Eskimo', ' Other'], 0, inplace = True)
    test_data.iloc[:,8].replace([' White', ' Asian-Pac-Islander'], 1, inplace = True)
    #sex
    test_data.iloc[:,9].replace([' Female'], 0, inplace = True)
    test_data.iloc[:,9].replace([' Male'], 1, inplace = True)
    #==============================================
    #drop the features we don't use : 11:'capital_loss' and 13:'native_country
    test_data.drop(labels=[13], axis = 1, inplace = True)
    test_X = test_data.values
    
    test_Y_pred = train_model.predict(test_X) 
    test_Y_pred = (test_Y_pred > 0.5).astype(int)
    #pd_data = pd.DataFrame(test_Y_pred, columns=['ans'])
    #pd_data.to_csv('answer.csv')

    #np.savetxt("answer.csv", output_arr, fmt="%d", delimiter=",")

    with open('answer1.csv','a',  newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['ID', 'ans'])
        a = list(range(0,len(test_Y_pred)))
        b = test_Y_pred.tolist()
        c = np.vstack((a,b))
        output_arr = np.array(c).transpose()
        for arr in output_arr:     
            writer.writerow(arr)
    f.close()    
if __name__ == "__main__":
    train_file = sys.argv[1];
    test_file = sys.argv[2];
#    train_file = "train.csv"
#    test_file = "test.csv"
    income_pred(train_file, test_file);


