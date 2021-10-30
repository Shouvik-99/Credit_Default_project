import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
cd=pd.read_csv("C:/Users/user/Desktop/ML_Project_Deploy/credit_default.csv")
cd.shape
cd.info()
x=cd.isnull().sum()/cd.shape[0]
x
cd=cd.drop(["ID"],axis=1)
cd.rename(columns={"PAY_0":"PAY_1"},inplace=True)
#cd["PAY_1"]=cd["PAY_1"].replace([0,-1,-2],0)
#cd["PAY_2"]=cd["PAY_2"].replace([0,-1,-2],0)
cd["PAY_1"][cd["PAY_1"]<=0]=0
cd["PAY_2"][cd["PAY_2"]<=0]=0
cd["PAY_3"][cd["PAY_3"]<=0]=0
cd["PAY_4"][cd["PAY_4"]<=0]=0
cd["PAY_5"][cd["PAY_5"]<=0]=0
cd["PAY_6"][cd["PAY_6"]<=0]=0
g = sns.pairplot(cd,x_vars=["LIMIT_BAL"],y_vars=["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","LIMIT_BAL"],diag_kind="kde",hue="default.payment.next.month")
#cd["PAY_1"][cd["PAY_1"]<=0]=0
cd["PAY_1"].value_counts()
g1 = sns.pairplot(cd,x_vars=["LIMIT_BAL"],y_vars=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],hue="default.payment.next.month")
g2= sns.pairplot(cd,x_vars=["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"],y_vars=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],hue="default.payment.next.month")
#g3= sns.pairplot(cd,x_vars=["BILL_AMT2"],y_vars=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"],hue="default.payment.next.month")
from scipy.stats import chi2_contingency
chi2,p1,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_1,cd["default.payment.next.month"]))
chi2,p2,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_2,cd["default.payment.next.month"]))
chi2,p3,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_3,cd["default.payment.next.month"]))
chi2,p4,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_4,cd["default.payment.next.month"]))
chi2,p5,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_5,cd["default.payment.next.month"]))
chi2,p6,dof,exp=chi2_contingency(pd.crosstab(cd.PAY_1,cd["default.payment.next.month"]))
p=[p1,p2,p3,p4,p5,p6]
col=["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
for i in zip(p,col):
    print(i)
d1=pd.DataFrame([cd["LIMIT_BAL"],cd["PAY_1"],cd["PAY_2"],cd["PAY_3"],cd["PAY_4"],cd["PAY_5"],cd["PAY_6"],cd["BILL_AMT1"],cd["BILL_AMT2"],cd["BILL_AMT3"],cd["BILL_AMT4"],cd["BILL_AMT5"],cd["BILL_AMT6"],cd["PAY_AMT1"],cd["PAY_AMT2"],cd["PAY_AMT3"],cd["PAY_AMT4"],cd["PAY_AMT5"],cd["PAY_AMT6"]]).transpose()
collist=["LIMIT_BAL","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
cd1=cd[collist]
np.set_printoptions(suppress=True,precision=6)
y=cd["default.payment.next.month"]
model=linear_model.LogisticRegression(penalty="l1",C=1000)
model.fit(cd1,y)
ser=pd.Series(model.coef_[0])
ser.index=cd1.columns
ser.abs().sort_values(ascending=False)

collist=["LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
cd1=cd[collist]
model=linear_model.LogisticRegression(penalty="l1",C=1000)
model.fit(cd1,y)
ser=pd.Series(model.coef_[0])
ser.index=cd1.columns
ser.abs().sort_values(ascending=False)

np.set_printoptions(suppress=True,precision=6)
X=cd.drop("default.payment.next.month",axis=1)
y=cd["default.payment.next.month"]
from sklearn import model_selection
from sklearn import tree
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import ensemble
cd_modify=cd.drop(["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","AGE"],axis=1)
X=cd_modify.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=37)
#Testing modelin whole dataset
def modelstats1(Xtrain,Xtest,ytrain,ytest):
    stats=[]
    modelnames=["LR","DecisionTree","KNN","NB"]
    models=list()
    models.append(linear_model.LogisticRegression())
    models.append(tree.DecisionTreeClassifier())
    models.append(neighbors.KNeighborsClassifier())
    models.append(naive_bayes.GaussianNB())
    for name,model in zip(modelnames,models):
        if name=="KNN":
            k=[l for l in range(5,17,2)]
            grid={"n_neighbors":k}
            grid_obj =model_selection.GridSearchCV(estimator=model,param_grid=grid,scoring="f1")
            grid_fit =grid_obj.fit(Xtrain,ytrain)
            model = grid_fit.best_estimator_
            model.fit(Xtrain,ytrain)
            name=name+"("+str(grid_fit.best_params_["n_neighbors"])+")"
            print(grid_fit.best_params_)
        else:
            model.fit(Xtrain,ytrain)
        trainprediction=model.predict(Xtrain)
        testprediction=model.predict(Xtest)
        scores=list()
        scores.append(name+"-train")
        scores.append(metrics.accuracy_score(ytrain,trainprediction))
        scores.append(metrics.precision_score(ytrain,trainprediction))
        scores.append(metrics.recall_score(ytrain,trainprediction))
        scores.append(metrics.roc_auc_score(ytrain,trainprediction))
        stats.append(scores)
        scores=list()
        scores.append(name+"-test")
        scores.append(metrics.accuracy_score(ytest,testprediction))
        scores.append(metrics.precision_score(ytest,testprediction))
        scores.append(metrics.recall_score(ytest,testprediction))
        scores.append(metrics.roc_auc_score(ytest,testprediction))
        stats.append(scores)
    
    colnames=["MODELNAME","ACCURACY","PRECISION","RECALL","AUC"]
    return pd.DataFrame(stats,columns=colnames)
modelstats1(Xtrain,Xtest,ytrain,ytest)
collist=["SEX","MARRIAGE","EDUCATION","AGE"]
cd2=cd[collist]
model=linear_model.LogisticRegression(penalty="l1",C=1000)
model.fit(cd2,y)
ser=pd.Series(model.coef_[0])
ser.index=cd2.columns
ser.abs().sort_values(ascending=False)
collist_modify=["SEX","MARRIAGE","EDUCATION","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","LIMIT_BAL","default.payment.next.month"]
Xtrain.columns
cd3=cd[collist_modify]
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=0)
collist_modify=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","LIMIT_BAL","default.payment.next.month"]
cd3=cd[collist_modify]
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=0)
collist_modify=["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","LIMIT_BAL","default.payment.next.month"]
cd3=cd[collist_modify]
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=0)
collist_modify=["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6","LIMIT_BAL","default.payment.next.month"]
cd3=cd[collist_modify]
cd3.columns
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=0)
collist_modify=["PAY_AMT1","PAY_AMT2","default.payment.next.month"]
cd3=cd[collist_modify]
cd3.columns
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.15,random_state=0)
X=cd.drop("default.payment.next.month",axis=1)
y=cd["default.payment.next.month"]
model=linear_model.LogisticRegression(penalty="l1",C=1000)
model.fit(X,y)
ser=pd.Series(model.coef_[0])
ser.index=X.columns
ser.abs().sort_values(ascending=False)
collist_modify=["PAY_1","PAY_3","PAY_6","PAY_2","PAY_AMT1","PAY_AMT2","PAY_AMT3","BILL_AMT1","default.payment.next.month"]
cd3=cd[collist_modify]
X=cd3.drop("default.payment.next.month",axis=1)
y=cd_modify["default.payment.next.month"]
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X,y,test_size=.1,random_state=0)
model=ensemble.RandomForestClassifier(n_estimators=2000)
model.fit(Xtrain,ytrain)
testprediction=model.predict(Xtest)
metrics.recall_score(ytest,testprediction)
metrics.roc_auc_score(ytest,testprediction)
cd["default.payment.next.month"].value_counts()
x={"0's frequncy":23364,"1's frequency":6636}
plt.bar(list(x.keys()), x.values(), color='b',width=.5)
