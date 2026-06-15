import pandas as pd
data_root='data\\data_500_pro.csv'
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

data_df_drop.reset_index(inplace=True)

#聚类分析，仅使用数值特征
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成一些随机的三维数据
n_samples = len(data_df_drop)
n_clusters = 1
# X, _ = make_blobs(n_samples=n_samples,n_features=3, centers=n_clusters, cluster_std=0.60, random_state=0)
X=data_df_drop[['filler', 'thickness（mm）', 'EMI_SE（SE/dB）']].values


# 可视化生成的数据
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o')
# ax.set_xlabel('filler')
# ax.set_ylabel('thickness(mm)')
# ax.set_zlabel('EMI_SE(SE/dB)')
# plt.title('3D Scatter Plot of Data Points')
# plt.show()

indices_to_drop=[]
centroids=None
def drop_point(X,indices_to_drop):
    indices_to_drop.clear()
# 使用K均值算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters,n_init='auto')
    kmeans.fit(X)


    # 获取聚类中心和预测的标签
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_


    num_points_to_remove = 1

    
    for cluster_idx in range(n_clusters):
        # 找出属于当前聚类的所有点的索引
        cluster_indices = np.where(labels == cluster_idx)[0]

        if X[cluster_indices[0]][2] >50 :
            continue  # 聚类中的点数少于等于要移除的点数，跳过处理

        # 计算每个点到聚类中心的距离
        distances_to_centroid = np.linalg.norm(X[cluster_indices] - centroids[cluster_idx], axis=1)
        # 按距离排序，获取距离最近的点的索引
        sorted_indices = cluster_indices[np.argsort(distances_to_centroid)]
        # 选择保留的点，即排除最远的几个点
        indices_to_drop.extend(sorted_indices[-num_points_to_remove:])
        
    labels=kmeans.labels_    
    return X,indices_to_drop,labels,centroids

drop_epcho=100
data_df_drop_2=data_df_drop.copy()
for i in range(drop_epcho):
    
    X,indices_to_drop,labels,centroids=drop_point(X,indices_to_drop)
    # print(indices_to_drop)
    data_df_drop_2.drop(labels=indices_to_drop,axis=0,inplace=True)
    
    if i != drop_epcho-1:
        data_df_drop_2.reset_index(inplace=True,drop=True)
        X=data_df_drop_2[['filler', 'thickness（mm）', 'EMI_SE（SE/dB）']].values
    # X,indices_to_drop,labels=drop_point(X,indices_to_drop)
# X=data_df_drop_2[['filler', 'thickness（mm）', 'EMI_SE（SE/dB）']].values
# kmeans = KMeans(n_clusters=n_clusters,n_init='auto')
# kmeans.fit(X)
# labels=kmeans.labels_


# 创建一个空列表来存储每个类别的数据
cluster_data = [[] for _ in range(n_clusters)]


# 根据标签将数据分组
for i, label in enumerate(labels):
    cluster_data[label].append(X[i])

# 将每个类别的数据转换为numpy数组，方便处理和保存
cluster_data = [np.array(cluster) for cluster in cluster_data]

# 输出每一类的数据
cluster_data_sum=0
for i, cluster in enumerate(cluster_data):
    cluster_data_sum+=len(cluster)
print("参与聚类数据个数:",cluster_data_sum)



# 可视化聚类结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, marker='o')
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200, label='Centroids')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.title('3D Scatter Plot with Clusters')
plt.legend()
plt.show()

data_df_drop_2.reset_index(inplace=True)
# data_df_drop_2

data_df_drop_2_copy=data_df_drop_2.copy()
data_df_drop_2_copy['emi_cat']=0

for idx, item in enumerate(data_df_drop_2_copy['EMI_SE（SE/dB）']):
    if item <20:
        data_df_drop_2_copy.at[idx,'emi_cat']=0
    if item > 20 and item <50:
        data_df_drop_2_copy.at[idx,'emi_cat']=1
    if item >50:
        data_df_drop_2_copy.at[idx,'emi_cat']=2
    
# data_df_drop_2_copy        

from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression


# X=data_df_drop_2_copy.drop(labels=['EMI_SE（SE/dB）','emi_cat'],axis=1).values
X=data_df_drop_2_copy[['filler', 'thickness（mm）']].values
Y=data_df_drop_2_copy['emi_cat'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
logi_mode=LogisticRegression(random_state=42,solver='lbfgs',max_iter=10000,multi_class='auto',)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
rf_score=rf_model.score(X_test, y_test)
gb_score=gb_model.score(X_test,y_test)

logi_mode.fit(X_train, y_train)

print("随机森林模型,rf_model.score:", rf_score)
print("梯度提升树模,gb_model.score:", gb_score)

y_pred = logi_mode.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Logi Accuracy: {accuracy:.2f}")

# 输出分类报告
# print(classification_report(y_test, y_pred))