from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() 
csr_mat = tfidf.fit_transform(documents)
print(csr_mat.toarray())
words = tfidf.get_feature_names()
print(words)

###TruncatedSVD == PCA for Sparse Matrixes
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)


import pandas as pd

pipeline.fit(articles)
labels = pipeline.predict(articles)
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))
