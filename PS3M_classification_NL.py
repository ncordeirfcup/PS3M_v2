import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from testset_prediction import testset_prediction as tsp
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
#from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
#import tkinter.font as font
from sklearn.feature_selection import VarianceThreshold
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

initialdir=os.getcwd()


def data1():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
    global col1
    col1 = list(file1.head(0))
    
def data2():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    
def data3():
    global filename3
    filename3 = askopenfilename(initialdir=initialdir,title = "Select validation file")
    fifthEntryTabOne.delete(0, END)
    fifthEntryTabOne.insert(0, filename3)
    global e_
    e_,f_=os.path.splitext(filename3)
    global file3
    file3 = pd.read_csv(filename3)
    
def data4():
    global filename4
    filename4 = askopenfilename(initialdir=initialdir,title = "Select classification model")
    sixthEntryTabOne.delete(0, END)
    sixthEntryTabOne.insert(0, filename4)
    #global e_
    #e_,f_=os.path.splitext(filename4)
    global file4
    file4 = pd.read_csv(filename4)

def data5():
    global filename5
    filename5 = askopenfilename(initialdir=initialdir,title = "Select model (.pkl) file")
    thirdEntryTabThree.delete(0, END)
    thirdEntryTabThree.insert(0, filename5)
    global clf
    clf=pickle.load(open(filename5, 'rb'))


def euclidean(x,y):  
    dist = np.sqrt(np.sum([(a-b)*(a-b) for a, b in zip(x, y)]))
    return round(dist,3)

def cal(dfX,dfmX,i,nd):
    lt,lt2,lt3=[],[],[]
    for n in dfX.columns:
        lt.append(euclidean(dfX[dfmX.columns[i]],dfX[n]))
        lt2.append(n)
        dfs=pd.DataFrame(zip(lt,lt2))
        mdata=dfs.sort_values(by=dfs.columns[0])
        mdata.columns=['ED','Descriptors']
    #print(list(mdata.iloc[0:nd,:].Descriptors.values))
    return list(mdata.iloc[0:nd,:].Descriptors.values),mdata.iloc[0:nd,:].ED

def corr(df):
    lt=[]
    df1=df.iloc[:,0:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    return max(flat_list),min(flat_list)


def correlation(X,cthreshold):
        col_corr = set() # Set of all the names of deleted columns
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (abs(corr_matrix.iloc[i, j]) > float(cthreshold)) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i] # getting the name of column
                    col_corr.add(colname)
                    if colname in X.columns:
                        del X[colname] # deleting the column from the dataset
        return X   

def pretreat(X,cthreshold):
    X=correlation(X,cthreshold)
    return X

def submit():
    nd=int(thirdEntryTabThreer3c1.get())
    dfmX=file4.iloc[:,1:nd+1]
    dfmy=file4.iloc[:,nd+1:nd+2]
    df1=file1
    df2=file2
    df3=file3
    #df=pd.concat([df1,df2], axis=0)
    #clf=LinearDiscriminantAnalysis()
    nx=int(thirdEntryTabThreer3c1_x.get())+1
    df1X1=df1.iloc[:,2:]
    df1X=pretreat(df1X1,0.999)
    df1y=df1.iloc[:,1:2]
    df2y=df2[df1y.columns]
    df3y=df3[df1y.columns]
    lt4,lt5,lt6,lt7=[],[],[],[]
    #dfmX=df1[['VE1sign_Dz(v)_cme', 'J_Dz(p)_cme', 'HATS5s_cbt', 'VE3sign_B(v)_cbt', 'Wap_cme', 'Eig09_EA(dm)_cme', 'DLS_01_cbt', 'nOHp_cme']]
    ##########################################################
    for i in range(dfmX.shape[1]):
        for j,k in zip(cal(df1X,dfmX,i,nx)[0][1:],cal(df1X,dfmX,i,nx)[1][1:]):
            lt7.append(dfmX.columns[i])
            try:
               nd=pd.concat([df1X[dfmX.columns],df1X[j]],axis=1).drop(dfmX[cal(df1X,dfmX,i,nx)[0][0:1]],axis=1)
            except KeyError:
                   messagebox.showinfo('Message','Highly collinear (r=1) descriptors was found in the sub-training set file, data pre-treatment is required')
            lt5.append(list(nd.columns)) 
            clf.fit(nd,df1.iloc[:,1:2])
            lt4.append(clf.score(nd,df1.iloc[:,1:2])) 
            lt6.append(k)
    dfs2=pd.DataFrame(zip(lt4,lt5,lt6,lt7))
    dfs2.columns=['Accuracy','Descriptors','ED','Replaced_with']
    dfs2.to_csv('ED_information_classification.csv',index=False)

       
    
    s1,i1,l1,l2,l3,l4=[],[],[],[],[],[]
    m1,m2,m3,m4,m5,m6,m51=[],[],[],[],[],[],[]
    m5_1, m51_1=[],[]
    lm, lm2=[],[]
    s1.append(list(dfmX.columns))
    clf.fit(df1X[dfmX.columns],df1y)
    ts=tsp(df1X[dfmX.columns],df1y,clf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_CV_GBM = cross_val_predict(clf, df1X[dfmX.columns], df1y, cv=cv)
    a7_1=(accuracy_score(df1y,y_pred_CV_GBM))*100
    a9_1=matthews_corrcoef(df1y,y_pred_CV_GBM)
    #print(a9_1)
    confusion_matrix_CV_GBM = confusion_matrix(df1y,y_pred_CV_GBM, labels=[0,1])  
    TN, FP, FN, TP = confusion_matrix_CV_GBM.ravel()
    a1_1=TP+TN
    #a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=ts.fit()
    i1.append(a7_1)
    tb1=dfmX.corr()
    mx1,mn1=corr(tb1)
    m6.append(round(max(mx1,abs(mn1)),3))
    ts=tsp(df2[dfmX.columns],df2y,clf)
    a11,a12,a13,a14,a15,a16,a17,a18,a19,a20=ts.fit()
    l1.append(a17)
    ts=tsp(df3[dfmX.columns],df3y,clf)
    a21,a22,a23,a24,a25,a26,a27,a28,a29,a30=ts.fit()
    l2.append(a27)
    l3.append((a7_1+a17+a27)/3)
    m1.append(a9_1)
    m2.append(a19)
    m3.append(a29)
    m4.append((a9_1+a19+a29)/3)
    m51.append(a1_1+a11+a12+a21+a22+((a9_1+a19+a29)/3))
    m51_1.append(a1_1+a9_1)
    m5.append(a1_1+a11+a12+a21+a22)
    m5_1.append(a1_1)
    l4.append('Original model')
    print(len(s1), len(i1), len(l1), len(l2), len(l3), len(m1), len(m2), len(m3), len(m4), len(m5_1), len(m51_1), len(m5), len(m51), len(m6), len(l4))
    Dict=dict([('Descriptors', s1),('Train_ACC', i1),('Test_ACC', l1), ('Validation_ACC', l2),('Average_ACC', l3),
            ('Train_MCC', m1),('Test_MCC', m2), ('Validation_MCC', m3),('Average_MCC',m4),('Correct_pred_train', m5_1),('CP_AMCC_train', m51_1),
            ('Correct_pred_total', m5),('CP_AMCC_total', m51),('Max_Inc',m6),('Model', l4)]) 
    table1=pd.DataFrame(Dict)
    for i in range(dfs2.sort_values('Accuracy',ascending=False).iloc[0:,:].shape[0]):
        df1X_1=df1X[list(dfs2.sort_values('Accuracy',ascending=False).iloc[i:i+1,:].Descriptors)[0]]
        df2X=df2[df1X_1.columns]
        df3X=df3[df1X_1.columns]
        clf.fit(df1X_1,df1y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_CV_GBM = cross_val_predict(clf, df1X_1, df1y, cv=cv)
        a7_1=(accuracy_score(df1y,y_pred_CV_GBM))*100
        a9_1=matthews_corrcoef(df1y,y_pred_CV_GBM)
        print(df1X_1.columns)
        print(a9_1)
        confusion_matrix_CV_GBM = confusion_matrix(df1y,y_pred_CV_GBM, labels=[0,1])  
        TN, FP, FN, TP = confusion_matrix_CV_GBM.ravel()
        a1_1=TP+TN
        tb1=dfmX.corr()
        mx1,mn1=corr(tb1)
        m6.append(round(max(mx1,abs(mn1)),3))
        ts=tsp(df2[df2X.columns],df2y,clf)
        a11,a12,a13,a14,a15,a16,a17,a18,a19,a20=ts.fit()
        s1.append(list(dfs2.sort_values('Accuracy',ascending=False).iloc[i:i+1,:].Descriptors)[0])
        i1.append(a7_1)
        l1.append(a17)
        ts=tsp(df3[df3X.columns],df3y,clf)
        a21,a22,a23,a24,a25,a26,a27,a28,a29,a30=ts.fit()
        l2.append(a27)
        l3.append((a7_1+a17+a27)/3)
        m1.append(a9_1)
        m2.append(a19)
        m3.append(a29)
        m4.append((a9_1+a19+a29)/3)
        m51.append(a1_1+a11+a12+a21+a22+((a9_1+a19+a29)/3))
        m51_1.append(a1_1+a9_1)
        m5.append(a1_1+a11+a12+a21+a22)
        m5_1.append(a1_1)
        l4.append('New model')
        
        Dict=dict([('Descriptors', s1),('Train_ACC', i1),('Test_ACC', l1), ('Validation_ACC', l2),('Average_ACC', l3),
            ('Train_MCC', m1),('Test_MCC', m2), ('Validation_MCC', m3),('Average_MCC',m4),('Correct_pred_train', m5_1),('CP_AMCC_train', m51_1),
            ('Correct_pred_total', m5),('CP_AMCC_total', m51),('Max_Inc',m6),('Model', l4)]) 
        
        table2=pd.DataFrame(Dict)
    
    if Criterion4.get()=='wl':
       dp='1-Wilks lambda'
       dp2='1-Wilks lambda'
    elif Criterion4.get()=='am':
       dp='Average_MCC'
       dp2='Train_MCC'
    elif Criterion4.get()=='fp':
       dp='Correct_pred_total'
       dp2='Correct_pred_train'
    elif Criterion4.get()=='cpam':
       dp='CP_AMCC_total'
       dp2='CP_AMCC_train'
    Table=table2.sort_values(by=dp,ascending=False)   
    if Table.iloc[0,:][dp2]>=Table.sort_values(by='Model',ascending=False).iloc[0,:][dp2] and Table.iloc[0,:][dp]>Table.sort_values(by='Model',ascending=False).iloc[0,:][dp]:
       messagebox.showinfo('Message','At least one better model was found, the best model saved')
       X=df1[Table.iloc[0,:].Descriptors]
       clf.fit(X,df1y)
       #clf.score(X,df1y)
       ypr=pd.DataFrame(clf.predict(X))
       ypr.columns=['Predict']
       model_train=pd.concat([df1.iloc[:,0:1],X,df1y,ypr],axis=1)
       tb=X.corr()
       mx,mn=corr(tb)
       tbn='best_model_corr.csv'
       tb.to_csv(tbn)
       #####Write text file for subtrain#########
       filer = open("best_model.txt","w")
       filer.write("Sub-training set results "+"\n")
       filer.write('Descriptors are: '+str(list(X.columns))+"\n")
       filer.write('Maximum correlation (Pearson r) between descriptors: '+str(round(max(mx,abs(mn)),3))+"\n")
       #filer.write("intercept: "+str(clf.intercept_)+"\n")
       #filer.write("coefficients: "+str(clf.coef_)+"\n")
       filer.write("\n")
       ts=tsp(X,df1y,clf)
       b1,b2,b3,b4,b5,b6,b7,b8,b9,b10=ts.fit()
       filer.write('True Positive: '+str(b1)+"\n")
       filer.write('True Negative: '+str(b2)+"\n")
       filer.write('False Positive '+str(b3)+"\n")
       filer.write('False Negative '+str(b4)+"\n")
       filer.write('Sensitivity: '+str(b5)+"\n")
       filer.write('Specificity: '+str(b6)+"\n")
       filer.write('Accuracy: '+str(b7)+"\n")
       filer.write('f1_score: '+str(b8)+"\n")
       #filer.write('Recall score: '+str(recall_score(self.y,ypred))
       filer.write('MCC: '+str(b9)+"\n")
       filer.write('ROC_AUC: '+str(b10)+"\n")
       filer.write('\n')
       ####
       model_train.to_csv('best_model_train.csv',index=False)
       yprts=pd.DataFrame(clf.predict(df2[X.columns]))
       yprts.columns=['Predict']
       model_test=pd.concat([df2.iloc[:,0:1],df2[X.columns],df2y,yprts],axis=1)
       model_test.to_csv('best_model_test.csv',index=False)
       #####Write text file for test#########
       ts=tsp(df2[X.columns],df2y,clf)
       b11,b12,b13,b14,b15,b16,b17,b18,b19,b20=ts.fit()
       filer.write("Test set results "+"\n")
       filer.write('True Positive: '+str(b11)+"\n")
       filer.write('True Negative: '+str(b12)+"\n")
       filer.write('False Positive '+str(b13)+"\n")
       filer.write('False Negative '+str(b14)+"\n")
       filer.write('Sensitivity: '+str(b15)+"\n")
       filer.write('Specificity: '+str(b16)+"\n")
       filer.write('Accuracy: '+str(b17)+"\n")
       filer.write('f1_score: '+str(b18)+"\n")
       #filer.write('Recall score: '+str(recall_score(self.y,ypred))
       filer.write('MCC: '+str(b19)+"\n")
       filer.write('ROC_AUC: '+str(b20)+"\n")
       filer.write('\n')
       #####Write validation file for test#########
       yprvd=pd.DataFrame(clf.predict(df3[X.columns]))
       yprvd.columns=['Predict']
       model_test=pd.concat([df3.iloc[:,0:1],df3[X.columns],df3y,yprvd],axis=1)
       model_test.to_csv('best_model_validation.csv',index=False)
       ts=tsp(df3[X.columns],df3y,clf)
       b21,b22,b23,b24,b25,b26,b27,b28,b29,b30=ts.fit()
       filer.write("Validation set results "+"\n")
       filer.write('True Positive: '+str(b21)+"\n")
       filer.write('True Negative: '+str(b22)+"\n")
       filer.write('False Positive '+str(b23)+"\n")
       filer.write('False Negative '+str(b24)+"\n")
       filer.write('Sensitivity: '+str(b25)+"\n")
       filer.write('Specificity: '+str(b26)+"\n")
       filer.write('Accuracy: '+str(b27)+"\n")
       filer.write('f1_score: '+str(b28)+"\n")
       #filer.write('Recall score: '+str(recall_score(self.y,ypred))
       filer.write('MCC: '+str(b29)+"\n")
       filer.write('ROC_AUC: '+str(b30)+"\n")
       filer.write('\n')
    elif Table.iloc[0,:][dp2]<Table.sort_values(by='Model',ascending=False).iloc[0,:][dp2] and Table.iloc[0,:][dp]>Table.sort_values(by='Model',ascending=False).iloc[0,:][dp]:
        messagebox.showinfo('Message','Overall predicivity improved but internal predictivity dropped, No model is saved')
    else:
       messagebox.showinfo('Message','No better model was found')
    #Table_m=pd.concat([table1,Table],axis=0)
    Table.to_csv('Table_classification_statistics.csv',index=False)
    
############Tab1########
form = tk.Tk()
form.title("PS3M_classification_forNL")
form.geometry("670x360")
tab_parent = ttk.Notebook(form)
tab1 = tk.Frame(tab_parent) #background='#ffffff')
tab_parent.add(tab1, text="PS3M_forNL")


firstLabelTabThree = tk.Label(tab1, text="Select complete sub-training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=65,y=10)
firstEntryTabThree = tk.Entry(tab1, width=40)
firstEntryTabThree.place(x=300,y=13)
b3=tk.Button(tab1,text='Browse', command=data1,font=("Helvetica", 10))
b3.place(x=550,y=10)  

secondLabelTabThree = tk.Label(tab1, text="Select complete test set",font=("Helvetica", 12))
secondLabelTabThree.place(x=120,y=40)
secondEntryTabThree = tk.Entry(tab1,width=40)
secondEntryTabThree.place(x=300,y=43)
b4=tk.Button(tab1,text='Browse', command=data2,font=("Helvetica", 10))
b4.place(x=550,y=40)

fifthLabelTabOne = tk.Label(tab1, text="Select complete validation set",font=("Helvetica", 12))
fifthLabelTabOne.place(x=77,y=70)
fifthEntryTabOne = tk.Entry(tab1, width=40)
fifthEntryTabOne.place(x=300,y=73)
b5=tk.Button(tab1,text='Browse', command=data3,font=("Helvetica", 10))
b5.place(x=550,y=70)  

sixthLabelTabOne = tk.Label(tab1, text="Select original classification model",font=("Helvetica", 12))
sixthLabelTabOne.place(x=48,y=100)
sixthEntryTabOne = tk.Entry(tab1, width=40)
sixthEntryTabOne.place(x=300,y=103)
b6=tk.Button(tab1,text='Browse', command=data4,font=("Helvetica", 10))
b6.place(x=550,y=100) 


thirdLabelTabThree = tk.Label(tab1, text="Select model (.pkl) file",font=("Helvetica", 12))
thirdLabelTabThree.place(x=50,y=130)
thirdEntryTabThree = tk.Entry(tab1,width=40)
thirdEntryTabThree.place(x=230,y=133)
b7=tk.Button(tab1,text='Browse', command=data5,font=("Helvetica", 10))
b7.place(x=480,y=130)


thirdLabelTabThreer2c1=Label(tab1, text='Number of descriptors in original model',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=45,y=160)
thirdEntryTabThreer3c1=Entry(tab1)
thirdEntryTabThreer3c1.place(x=350,y=163)

thirdLabelTabThreer2c1_x=Label(tab1, text='Number of descriptors with minimum ED',font=("Helvetica", 12))
thirdLabelTabThreer2c1_x.place(x=45,y=190)
thirdEntryTabThreer3c1_x=Entry(tab1)
thirdEntryTabThreer3c1_x.place(x=350,y=193)

Criterion_Label4 = ttk.Label(tab1, text="Parameter for selecting the best model",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion_Label4.place(x=250,y=210)
Criterion4 = StringVar()
Criterion4.set('fp')
Criterion_acc3 = ttk.Radiobutton(tab1, text='Wilks lambda', variable=Criterion4, value='wl')
#Criterion_roc3 = ttk.Radiobutton(tab1, text='Average accuracy', variable=Criterion4, value='aa')
Criterion_roc4 = ttk.Radiobutton(tab1, text='Average MCC', variable=Criterion4, value='am')
Criterion_roc5 = ttk.Radiobutton(tab1, text='#Correct predictions', variable=Criterion4, value='fp')
Criterion_roc5x = ttk.Radiobutton(tab1, text='#Correct predictions+Average MCC', variable=Criterion4, value='cpam')

Criterion_acc3.place(x=40,y=233)
#Criterion_roc3.place(x=100,y=233)
Criterion_roc4.place(x=160,y=233)
Criterion_roc5.place(x=270,y=233)
Criterion_roc5x.place(x=420,y=233)

b8=Button(tab1, text='Submit', command=submit,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b8.place(x=340,y=275)


tab_parent.pack(expand=1, fill='both')


form.mainloop()