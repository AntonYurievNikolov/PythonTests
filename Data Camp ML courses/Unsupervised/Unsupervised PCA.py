###PCA
from sklearn import datasets
iris = datasets.load_iris()

from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
width = iris.data[:,0]
length = iris.data[:,2]
plt.scatter(width, length)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(width, length)
print(correlation)
#Decorrelation
model = PCA()
pca_features = model.fit_transform(iris.data)
xs = pca_features[:,0]
ys = pca_features[:,2]
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()
correlation, pvalue = pearsonr(xs, ys)
print(correlation)
#Where does the data varies the most
plt.scatter(iris.data[:,0], iris.data[:,2])
model = PCA()
model.fit(iris.data)

mean = model.mean_
first_pc = model.components_[0,:]
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)
plt.axis('equal')
plt.show()

#Intrensic Dimension
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(iris.data)
# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

#The Actual Dim reduction
from sklearn.preprocessing import normalize
iris_scaled = normalize(iris.data)
pca = PCA(n_components=2)
pca.fit(iris_scaled)
pca_features = pca.transform(iris_scaled)
# Print the shape of pca_features
print(pca_features.shape)

