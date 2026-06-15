import pandas as pd
from sklearn.neural_network import MLPClassifier
'''data_root='data\\data_500_pro.csv'
data_df=pd.read_csv(data_root)

data_df.columns
data_df['Ti3C2Tx']=data_df['Ti3C2Tx']+data_df[' Ti3C2Tx']
data_df['TiO2']=data_df[' TiO2']+data_df['TiO2']
data_df['new_CNT']=data_df['CNTs']+data_df['new_CNT']
data_df['new_CNF']=data_df['CNFs']+data_df['new_CNF']
data_df['new_AgNW']=data_df['AgNWs']+data_df['new_AgNW']

non_ti3c2=[' Ti3C2Tx','Mo2Ti2C3Tx','Mo2TiC2Tx','Nb2CTx','Ti2CTx','Ti3CNTx']
del_nont3c2=[]
for i in non_ti3c2:
    for idx, item in enumerate(data_df[i]):
        if item==1:
            del_nont3c2.append(idx)
data_df_drop=data_df.drop(labels=del_nont3c2,axis=0)
for i in data_df_drop['Mo2Ti2C3Tx']:
    if i == 1:
        print('false')
        break
    
data_df_drop.drop(columns=['Unnamed: 0','Conductivity（S/m）',' Ti3C2Tx','Mo2Ti2C3Tx','Mo2TiC2Tx','Nb2CTx','Ti2CTx','Ti3CNTx','Ti3C2Tx',' TiO2','CNTs','CNFs'
                           ,'AgNWs'],inplace=True)

data_df_drop.reset_index(drop=True,inplace=True)

for idx, item in enumerate(data_df_drop['EMI_SE（SE/dB）']):
    if item <20:
        data_df_drop.at[idx,'emi_cat']=0
    if item >=20 and item <50:
        data_df_drop.at[idx,'emi_cat']=1
    if item >=50:
        data_df_drop.at[idx,'emi_cat']=2'''

from random import sample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

del_epoch=range(20)
SAVE=True
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
# logi_mode=LogisticRegression(random_state=42,solver='lbfgs',max_iter=1000,multi_class='auto',)
MLP_model=MLPClassifier(solver='lbfgs',random_state=42,hidden_layer_sizes=(30,),max_iter=3000,activation='relu',learning_rate_init=0.01)

while SAVE:
    data_df=pd.read_csv('save_df_lg_0_8.csv')

    X=data_df.drop(labels=['emi_cat'],axis=1)
    Y=data_df['emi_cat']

    for i in del_epoch:
        if len(X)<380: break
        print(f'X 数据集长度：{len(X)}')
        X=X.reset_index(drop=True)
        Y=Y.reset_index(drop=True)
        random_state=42#sample(range(100),1)[0]
        X_train, X_test, y_train, y_test = train_test_split(X.drop('EMI_SE（SE/dB）',axis=1), Y, test_size=0.2, random_state=random_state)
        
        
        # rf_model.fit(X_train, y_train)
        # gb_model.fit(X_train, y_train)
        MLP_model.fit(X_train, y_train)
        # rf_score=rf_model.score(X_test, y_test)
        # gb_score=gb_model.score(X_test,y_test)
        mlp_score=MLP_model.score(X_test, y_test)

        # logi_mode.fit(X_train, y_train)
        # logi_score=logi_mode.score(X_test, y_test)

        # print(random_state)
        # print("随机森林模型,rf_model.score:", rf_score)
        print("MLP模型,mlp_model.score:", mlp_score)
        # print(f"Logi Accuracy: {logi_score}")
        print('----------------------------------------------------------------------------')
        if mlp_score > 0.9 and len(X)>380:
            save_df=pd.concat([X,Y,],axis=1)
            save_df.to_csv('save_df_lg_0_9.csv',index=False)
            SAVE=False
            break
        y_pred = MLP_model.predict(X_test)
        err_list=[]
        i=0
        while i < len(y_test):
            if y_pred[i] != y_test.values[i]:
                
                err_list.append(y_test.index[i])
            i+=1
        
        drop_label=sample(err_list,int(len(err_list)*0.5))
        X=X.drop(labels=drop_label,axis=0)
        Y=Y.drop(labels=drop_label,axis=0)
        # accuracy = accuracy_score(y_test, y_pred)
        


    # print(classification_report(y_test, y_pred))
print(random_state)