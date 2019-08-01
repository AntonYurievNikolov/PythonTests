###Kmeans
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pandas as pd
from matplotlib import pyplot as plt

iris = datasets.load_iris()
ks = range(1, 6)
inertias = []

X = iris.data  # we only take the first two features.
y = iris.target

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
#Crosstab check
scaler = StandardScaler()
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, model)
labels = pipeline.fit_predict(X)
df = pd.DataFrame({'labels': labels, 'varieties': y})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

###Hieararchical Clustering
from scipy.cluster.hierarchy import linkage , dendrogram,fcluster

mergings = linkage(normalize(X),method='complete')
dendrogram(mergings,
           labels=y,
           leaf_rotation=90,
           leaf_font_size=6,
)
plt.show()

labels = fcluster(mergings, 6, criterion='distance')
df = pd.DataFrame({'labels': labels, 'varieties': y})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

#t-SNE
# Import TSNE
from sklearn.manifold import TSNE

model = TSNE(learning_rate=50)
tsne_features = model.fit_transform(X)
xs = tsne_features[:,0]
ys = tsne_features[:,1]
plt.scatter(xs, ys)
plt.show()

# Annotate the points
#for x, y, company in zip(xs, ys, y):
#    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
#plt.show()
