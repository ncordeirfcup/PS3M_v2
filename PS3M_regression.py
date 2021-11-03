import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
import numpy as np 
import os
import shutil
import numpy as np
from tkinter.filedialog import askopenfilename
#from tkinter.filedialog import askopenfilenames
import time
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LinearRegression
from loo import loo
from rm2 import rm2
from sklearn.metrics import mean_absolute_error,mean_squared_error
import statsmodels.api as sm

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
    filename4 = askopenfilename(initialdir=initialdir,title = "Select regression model")
    sixthEntryTabOne.delete(0, END)
    sixthEntryTabOne.insert(0, filename4)
    #global e_
    #e_,f_=os.path.splitext(filename4)
    global file4
    file4 = pd.read_csv(filename4)

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

def submit():
    nd=int(thirdEntryTabThreer3c1.get())
    dfmX=file4.iloc[:,1:nd+1]
    dfmy=file4.iloc[:,nd+1:nd+2]
    df1=file1
    df2=file2
    df3=file3
    #df=pd.concat([df1,df2], axis=0)
    reg=LinearRegression()
    nx=int(thirdEntryTabThreer3c1_x.get())+1
    df1X=df1.iloc[:,2:]
    df1y=df1.iloc[:,1:2]
    df2y=df2[df1y.columns]
    df3y=df3[df1y.columns]
    lt4,lt5=[],[]
    #dfmX=df1[['VE1sign_Dz(v)_cme', 'J_Dz(p)_cme', 'HATS5s_cbt', 'VE3sign_B(v)_cbt', 'Wap_cme', 'Eig09_EA(dm)_cme', 'DLS_01_cbt', 'nOHp_cme']]
    lt6,lt7=[],[]
    for i in range(dfmX.shape[1]):
        for j,k in zip(cal(df1X,dfmX,i,nx)[0][1:],cal(df1X,dfmX,i,nx)[1][1:]):
            lt7.append(dfmX.columns[i])
            try:
               nd=pd.concat([df1X[dfmX.columns],df1X[j]],axis=1).drop(dfmX[cal(df1X,dfmX,i,nx)[0][0:1]],axis=1)
            except KeyError:
               messagebox.showinfo('Message','Highly collinear (r=1) descriptors was found in the sub-training set file, data pre-treatment is required')
            lt5.append(list(nd.columns))
            reg.fit(nd,df1.iloc[:,1:2])
            lt4.append(reg.score(nd,df1.iloc[:,1:2]))
            lt6.append(k)
    dfs2=pd.DataFrame(zip(lt4,lt5,lt6,lt7))
    dfs2.columns=['R2','Descriptors','ED','Replaced_with']
    dfs2.to_csv('ED_information_regression.csv',index=False)
    s1,i1,l1,l2,l3,l4=[],[],[],[],[],[]
    m1,m2,m3,m4,m5,m6=[],[],[],[],[],[]
    d1,d2,d3=[],[],[]
    lm=[]
    s1.append(list(dfmX.columns))
    reg.fit(df1X[dfmX.columns],df1y)
    i1.append(reg.score(df1X[dfmX.columns],df1y)) ###r2
    cv=loo(df1X[dfmX.columns],df1y,file1)
    c,m,l=cv.cal()
    lm.append(c) ###q2loo
    ypr=pd.DataFrame(reg.predict(df1X[dfmX.columns]))
    ypr.columns=['Pred']
    rm2tr,drm2tr=rm2(df1y,l).fit()
    d=mean_absolute_error(df1y,ypr)
    #e=(mean_squared_error(df1y,ypr))**0.5
    l1.append(rm2tr)    ###rm2tr
    d1.append(drm2tr)   ###delta rm2tr
    #l2.append(drm2tr)
    l2.append(d)    ###MAE_tr
    #l4.append(e)
    tb1=dfmX.corr()
    mx1,mn1=corr(tb1)
    m6.append(round(max(mx1,abs(mn1)),3))
    #####original test set data
    ytspr=pd.DataFrame(reg.predict(df2[dfmX.columns]))
    ytspr.columns=['Pred']
    rm2ts,drm2ts=rm2(df2y,ytspr).fit()
    tsdf=pd.concat([df2y,pd.DataFrame(ytspr)],axis=1)
    tsdf.columns=['Active','Predict']
    tsdf['Aver']=m
    tsdf['Aver2']=tsdf['Predict'].mean()
    tsdf['diff']=tsdf['Active']-tsdf['Predict']
    tsdf['diff2']=tsdf['Active']-tsdf['Aver']
    tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
    r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
    r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
    RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
    l3.append(r2pr)      ###r2pred of test set
    #m2.append(r2pr2)
    l4.append(rm2ts)     ###rm2ts of test set
    d2.append(drm2ts)     ###delta delta rm2 of test set
    #m4.append(drm2ts)
    #####original validation set data
    ytspr_v=pd.DataFrame(reg.predict(df3[dfmX.columns]))
    ytspr_v.columns=['Pred']
    rm2ts_v,drm2ts_v=rm2(df3y,ytspr_v).fit()
    tsdf_v=pd.concat([df3y,pd.DataFrame(ytspr_v)],axis=1)
    tsdf_v.columns=['Active','Predict']
    tsdf_v['Aver']=m
    tsdf_v['Aver2']=tsdf_v['Predict'].mean()
    tsdf_v['diff']=tsdf_v['Active']-tsdf_v['Predict']
    tsdf_v['diff2']=tsdf_v['Active']-tsdf_v['Aver']
    tsdf_v['diff3']=tsdf_v['Active']-tsdf_v['Aver2']
    r2pr_v=1-((tsdf_v['diff']**2).sum()/(tsdf_v['diff2']**2).sum())
    r2pr2_v=1-((tsdf_v['diff']**2).sum()/(tsdf_v['diff3']**2).sum())
    RMSEP_v=((tsdf_v['diff']**2).sum()/tsdf_v.shape[0])**0.5
    m1.append(r2pr_v) ###r2pred of validation set
    m2.append(rm2ts_v)  ###rm2ts of validation set
    d3.append(drm2ts_v)     ###delta delta rm2 of validation set
    m3.append((c+r2pr+r2pr_v)/3)
    m4.append((rm2tr+rm2ts+rm2ts_v)/3)
    m5.append('Original model')
    Dict=dict([('Descriptors', s1),('R2',i1),('Q2LOO', lm),('rm2_tr', l1), ('drm2_tr',d1),('MAE_tr', l2),('R2Pred_ts', l3),
            ('rm2_ts', l4),('drm2_ts',d2),('R2Pred_vd', m1), ('rm2_vd', m2),('drm2_vd',d3),('Average_Pred',m3),('Average rm2', m4),('Max_Inc',m6),('Model', m5)])
    #table1=pd.DataFrame(Dict)
    for i in range(dfs2.sort_values('R2',ascending=False).iloc[0:,:].shape[0]):
        df1X_1=df1X[list(dfs2.sort_values('R2',ascending=False).iloc[i:i+1,:].Descriptors)[0]]
        df2X=df2[df1X_1.columns]
        df3X=df3[df1X_1.columns]
        reg.fit(df1X_1,df1y)
        s1.append(list(dfs2.sort_values('R2',ascending=False).iloc[i:i+1,:].Descriptors)[0])
        i1.append(reg.score(df1X_1,df1y))
        cv=loo(df1X_1,df1y,file1)
        c,m,l=cv.cal()
        lm.append(c) ###q2loo
        yprn=pd.DataFrame(reg.predict(df1X_1))
        yprn.columns=['Pred']
        rm2trn,drm2trn=rm2(df1y,l).fit()
        dn=mean_absolute_error(df1y,yprn)
        en=(mean_squared_error(df1y,yprn))**0.5
        l1.append(rm2trn)    ###rm2tr
        d1.append(drm2trn)   ###delta rm2tr
        #l2.append(drm2tr)
        l2.append(dn)    ###MAE_tr
        #l4.append(e)
        tb1n=df1X_1.corr()
        mx1n,mn1n=corr(tb1n)
        m6.append(round(max(mx1n,abs(mn1n)),3))
        #####original test set data
        ytsprn=pd.DataFrame(reg.predict(df2X))
        ytsprn.columns=['Pred']
        rm2tsn,drm2tsn=rm2(df2y,ytsprn).fit()
        tsdfn=pd.concat([df2y,pd.DataFrame(ytsprn)],axis=1)
        tsdfn.columns=['Active','Predict']
        tsdfn['Aver']=m
        tsdfn['Aver2']=tsdfn['Predict'].mean()
        tsdfn['diff']=tsdfn['Active']-tsdfn['Predict']
        tsdfn['diff2']=tsdfn['Active']-tsdfn['Aver']
        tsdfn['diff3']=tsdfn['Active']-tsdfn['Aver2']
        r2prn=1-((tsdfn['diff']**2).sum()/(tsdfn['diff2']**2).sum())
        r2pr2n=1-((tsdfn['diff']**2).sum()/(tsdfn['diff3']**2).sum())
        RMSEPn=((tsdfn['diff']**2).sum()/tsdfn.shape[0])**0.5
        l3.append(r2prn)      ###r2pred of test set
        #m2.append(r2pr2)
        l4.append(rm2tsn)     ###rm2ts of test set
        d2.append(drm2tsn)   ###delta rm2ts
        #m4.append(drm2ts)
        #####original validation set data
        ytspr_vn=pd.DataFrame(reg.predict(df3X))
        ytspr_vn.columns=['Pred']
        rm2ts_vn,drm2ts_vn=rm2(df3y,ytspr_vn).fit()
        tsdf_vn=pd.concat([df3y,pd.DataFrame(ytspr_vn)],axis=1)
        tsdf_vn.columns=['Active','Predict']
        tsdf_vn['Aver']=m
        tsdf_vn['Aver2']=tsdf_vn['Predict'].mean()
        tsdf_vn['diff']=tsdf_vn['Active']-tsdf_vn['Predict']
        tsdf_vn['diff2']=tsdf_vn['Active']-tsdf_vn['Aver']
        tsdf_vn['diff3']=tsdf_vn['Active']-tsdf_vn['Aver2']
        r2pr_vn=1-((tsdf_vn['diff']**2).sum()/(tsdf_vn['diff2']**2).sum())
        r2pr2_vn=1-((tsdf_vn['diff']**2).sum()/(tsdf_vn['diff3']**2).sum())
        RMSEP_vn=((tsdf_vn['diff']**2).sum()/tsdf_vn.shape[0])**0.5
        m1.append(r2pr_vn) ###r2pred of validation set
        m2.append(rm2ts_vn)  ###rm2ts of validation set
        d3.append(drm2ts_vn)   ###delta rm2vd
        m3.append((c+r2prn+r2pr_vn)/3)
        m4.append((rm2trn+rm2tsn+rm2ts_vn)/3)
        m5.append('New model')
        Dict=dict([('Descriptors', s1),('R2',i1),('Q2LOO', lm),('rm2_tr', l1), ('drm2_tr',d1),('MAE_tr', l2),('R2Pred_ts', l3),
              ('rm2_ts', l4),('drm2_ts',d2),('R2Pred_vd', m1), ('rm2_vd', m2),('drm2_vd',d3),('Average_Pred',m3),('Average rm2', m4),('Max_Inc',m6),('Model', m5)])
        table2=pd.DataFrame(Dict)
    
    if Criterion4.get()=='apr':
       dp='Average_Pred'
       dp2='Q2LOO'
    elif Criterion4.get()=='arm':
       dp='Average rm2'
       dp2='rm2_tr'
    Table=table2.sort_values(by=dp,ascending=False)
    if  Table.iloc[0,:][dp2]>=Table.sort_values(by='Model',ascending=False).iloc[0,:][dp2] and Table.iloc[0,:][dp]>Table.sort_values(by='Model',ascending=False).iloc[0,:][dp]:
       messagebox.showinfo('Message','At least one better model was found, the best model saved')
       X=df1[Table.iloc[0,:].Descriptors]
       reg.fit(X,df1y)
       ypr=pd.DataFrame(reg.predict(X))
       ypr.columns=['Predict']
       model_train=pd.concat([df1.iloc[:,0:1],X,df1y,ypr],axis=1)
       model_train.to_csv('best_model_train.csv',index=False)
       yprts=pd.DataFrame(reg.predict(df2[X.columns]))
       yprts.columns=['Predict']
       model_test=pd.concat([df2.iloc[:,0:1],df2[X.columns],df2y,yprts],axis=1)
       model_test.to_csv('best_model_test.csv',index=False)
       yprvd=pd.DataFrame(reg.predict(df3[X.columns]))
       yprvd.columns=['Predict']
       model_test=pd.concat([df3.iloc[:,0:1],df3[X.columns],df3y,yprvd],axis=1)
       model_test.to_csv('best_model_validation.csv',index=False)
    
       filer = open("best_reg_model.txt","w")
       filer.write("Sub-training set results "+"\n")
       filer.write('Descriptors are: '+str(list(X.columns))+"\n")
       filer.write('Maximum correlation (Pearson r) between descriptors: '+str(Table.iloc[0,:].Max_Inc)+"\n")
       model=sm.OLS(df1y, sm.add_constant(pd.DataFrame(X)))
       results=model.fit()
       b=results.summary()
       filer.write("Statistics:"+str(b)+"\n")
       filer.write('Training set results: '+"\n")
       filer.write('Maximum correlation (Pearson r) between descriptors: '+str(Table.iloc[0,:].Max_Inc)+"\n")
       filer.write('Q2LOO: '+str(Table.iloc[0,:].Q2LOO)+"\n")
       filer.write('MAE_tr: '+str(Table.iloc[0,:].MAE_tr)+"\n")
       filer.write('Rm2LOO: '+str(Table.iloc[0,:].rm2_tr)+"\n")
       filer.write('delta Rm2LOO: '+str(Table.iloc[0,:].drm2_tr)+"\n")
       reg.fit(X,df1y)
       ypr=pd.DataFrame(reg.predict(X))
       ypr.columns=['Pred']
       e=(mean_squared_error(df1y,ypr))**0.5
       filer.write('RMSE: '+str(e)+"\n")
       filer.write('\n')
       filer.write("Test set results "+"\n")
       filer.write('R2Pred: '+str(Table.iloc[0,:].R2Pred_ts)+"\n")
       filer.write('Rm2_ts: '+str(Table.iloc[0,:].rm2_ts)+"\n")
       filer.write('delta Rm2ts: '+str(Table.iloc[0,:].drm2_ts)+"\n")
       filer.write('\n')
       filer.write("Validation set results "+"\n")
       filer.write('R2Pred_vd: '+str(Table.iloc[0,:].R2Pred_vd)+"\n")
       filer.write('Rm2_vd: '+str(Table.iloc[0,:].rm2_vd)+"\n")
       filer.write('delta Rm2vd: '+str(Table.iloc[0,:].drm2_vd)+"\n")
    elif Table.iloc[0,:][dp2]<Table.sort_values(by='Model',ascending=False).iloc[0,:][dp2] and Table.iloc[0,:][dp]>Table.sort_values(by='Model',ascending=False).iloc[0,:][dp]:
        messagebox.showinfo('Message','Overall predicivity improved but internal predictivity dropped, No model is saved')
    else:
       messagebox.showinfo('Message','No better model was found')
    #Table_m=pd.concat([table1,Table],axis=0)
    Table.to_csv('Table_regression_statistics.csv',index=False)
    
############Tab1########
form = tk.Tk()
form.title("PS3M_regression")
form.geometry("670x310")
tab_parent = ttk.Notebook(form)
tab1 = tk.Frame(tab_parent) #background='#ffffff')
tab_parent.add(tab1, text="PS3M")


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

sixthLabelTabOne = tk.Label(tab1, text="Select original regression model",font=("Helvetica", 12))
sixthLabelTabOne.place(x=48,y=100)
sixthEntryTabOne = tk.Entry(tab1, width=40)
sixthEntryTabOne.place(x=300,y=103)
b6=tk.Button(tab1,text='Browse', command=data4,font=("Helvetica", 10))
b6.place(x=550,y=100) 

thirdLabelTabThreer2c1=Label(tab1, text='Number of descriptors in original model',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=45,y=130)
thirdEntryTabThreer3c1=Entry(tab1)
thirdEntryTabThreer3c1.place(x=350,y=133)

thirdLabelTabThreer2c1_x=Label(tab1, text='Number of descriptors with minimum ED',font=("Helvetica", 12))
thirdLabelTabThreer2c1_x.place(x=45,y=160)
thirdEntryTabThreer3c1_x=Entry(tab1)
thirdEntryTabThreer3c1_x.place(x=350,y=163)

Criterion_Label4 = ttk.Label(tab1, text="Parameter for selecting the best model",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion_Label4.place(x=230,y=190)
Criterion4 = StringVar()
Criterion4.set('arm')
Criterion_acc3 = ttk.Radiobutton(tab1, text='Average prediction (Q2LOO+average R2Pred)', variable=Criterion4, value='apr')
Criterion_roc4 = ttk.Radiobutton(tab1, text='Average rm2 statistics (rm2LOO+average rm2test)', variable=Criterion4, value='arm')

Criterion_acc3.place(x=60,y=223)
#Criterion_roc3.place(x=100,y=213)
Criterion_roc4.place(x=330,y=223)




b7=Button(tab1, text='Submit', command=submit,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b7.place(x=300,y=250)


tab_parent.pack(expand=1, fill='both')


form.mainloop()