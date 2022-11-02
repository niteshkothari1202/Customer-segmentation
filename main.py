import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
import joblib

import warnings
warnings.filterwarnings("ignore")
with open("Clustered_Customer_Data.csv", "r") as fp:
    df = pd.read_csv("Clustered_Customer_Data.csv", index_col="Unnamed: 0")

df = df.drop(['Cluster'], axis=1)
df["MINIMUM_PAYMENTS"] = df["MINIMUM_PAYMENTS"].fillna(df["MINIMUM_PAYMENTS"].mean())
df["CREDIT_LIMIT"] = df["CREDIT_LIMIT"].fillna(df["CREDIT_LIMIT"].mean())

plt.figure(figsize=(30, 45))
for i, col in enumerate(df.columns):
    if df[col].dtype != 'object':
        ax = plt.subplot(9, 2, i + 1)
        sns.kdeplot(df[col], ax=ax)
        plt.xlabel(col)
plt.savefig(f"graphs\graph_1.png")
#
plt.figure(figsize=(10, 60))
for i in range(0, 17):
    plt.subplot(17,1,i+1)
    sns.distplot(df[df.columns[i]],kde_kws={'color':'black','bw': 0.1,'lw':3,'label':'KDE'},hist_kws={'color':'g'})
    plt.title(df.columns[i])
plt.savefig("graphs\graph_2.png")

plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), annot=True)
plt.savefig("graphs\graph_heatmap.png")

scalar=StandardScaler()
scaled_df = scalar.fit_transform(df)
# principal component analysis
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=principal_components, columns=["PCA1","PCA2"])
print(pca_df)

# K value by elbow method

inertia = []
range_val = range(1, 15)
for i in range_val:
    kmean = KMeans(n_clusters=i)
    kmean.fit_predict(pd.DataFrame(scaled_df))
    inertia.append(kmean.inertia_)
plt.plot(range_val, inertia, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.savefig("graphs\Kmeans_cluster_prediction.png")

# Kmeans
kmeans_model=KMeans(4)
kmeans_model.fit_predict(scaled_df)
pca_df_kmeans = pd.concat([pca_df, pd.DataFrame({'cluster': kmeans_model.labels_})], axis=1)

# saving the plot
plt.figure(figsize=(8, 8))
ax=sns.scatterplot(x="PCA1", y="PCA2", hue="cluster", data=pca_df_kmeans, palette=['red', 'green', 'blue', 'black'])
plt.title("Clustering using K-Means Algorithm")
plt.savefig("graphs\cluster.png")

# Cluster centers
cluster_centers = pd.DataFrame(data=kmeans_model.cluster_centers_, columns=[df.columns])

# inverse transform of data ( back from pca)

cluster_centers = scalar.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data=cluster_centers, columns=[df.columns])
print(cluster_centers)

cluster_df = pd.concat([df, pd.DataFrame({'Cluster': kmeans_model.labels_})], axis=1)
print(cluster_df)

cluster_2_df = cluster_df[cluster_df["Cluster"] == 1]
print(cluster_2_df)

cluster_3_df = cluster_df[cluster_df["Cluster"] == 2]
print(cluster_3_df)

cluster_4_df = cluster_df[cluster_df["Cluster"] == 3]
print(cluster_4_df)

sns.countplot(x='Cluster', data=cluster_df)
plt.savefig("graphs\Cluster_count.png")

for c in cluster_df.drop(['Cluster'],axis=1):
     grid = sns.FacetGrid(cluster_df, col='Cluster')
     grid = grid.map(plt.hist, c)
plt.savefig("graphs\Cluster_parameter_relation.png")

#Saving Scikitlearn models
joblib.dump(kmeans_model, "kmeans_model.pkl")
cluster_df.to_csv("Clustered_Customer_Data.csv")

# Training and Testing the data
# Split Dataset
X = cluster_df.drop(['Cluster'],axis=1)
y = cluster_df[['Cluster']]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3,random_state=3)


# Decision Tree Classification

model= DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# confusion Matrix

matrix = metrics.confusion_matrix(y_test, y_pred)
reports = classification_report(y_test, y_pred)

with open("Decision_Tree_Reports","w") as f:
    f.write(reports)
matrix = pd.DataFrame(matrix)
matrix.to_csv("Decision_Tree_Matrix.csv")
