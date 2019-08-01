####NMF
import pylab as pl
from sklearn.decomposition import  NMF
from sklearn import datasets
from sklearn.preprocessing import normalize
import pandas as pd
digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target
n_features = X.shape[1]
n_components = 16
nmf = NMF(n_components=n_components).fit(X)
nmfdigits = nmf.components_

# Plot the results
n_row, n_col = 4, 4
f2 = pl.figure(figsize=(1. * n_col, 1.13 * n_row))
f2.text(.5, .95, 'Non-negative components', horizontalalignment='center')
for i in range(n_row * n_col):
    pl.subplot(n_row, n_col, i + 1)
    pl.imshow(nmfdigits[i].reshape((8, 8)), cmap=pl.cm.gray,
              interpolation='nearest')
    pl.xticks(())
    pl.yticks(())
pl.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)
pl.show()

###NMF to World Frequency Array for Recommendation
# =============================================================================
# features = nmf.fit_transform(X)
# norm_features = normalize(features)
# print(norm_features)
# df = pd.DataFrame(norm_features, index=y)
# d = df.iloc[1]
# similarities = df.dot(d)
# print(similarities.nlargest())
# =============================================================================


# =============================================================================
# from sklearn.decomposition import NMF
# from sklearn.preprocessing import Normalizer, MaxAbsScaler
# from sklearn.pipeline import make_pipeline
# 
# scaler = MaxAbsScaler()
# nmf = NMF(n_components=20)
# normalizer = Normalizer()
# pipeline = make_pipeline(scaler, nmf, normalizer)
# norm_features = pipeline.fit_transform(artists)
# 
# 
# import pandas as pd
# 
# df = pd.DataFrame(norm_features, index=artist_names)
# artist = df.loc['Bruce Springsteen']
# similarities = df.dot(artist)
# print(similarities.nlargest())
# =============================================================================
