import time

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from pipeline_grid_search import PipelineGridSearchCV

pipe = Pipeline([
    ("fu", FeatureUnion([
        ("feat1", Pipeline([
            ("kbest", SelectKBest()),
            ])),
        ("feat2", Pipeline([
            ("pca", PCA()),
            ("norm", Normalizer())
            ])),
        ])),
    ("kbest", SelectKBest()),
    ("svm", SVC()),
    ])

cv_params = dict([
    ('fu__feat1__kbest__k', [1,2,3,4,5]),
    ('fu__feat2__pca__n_components', [100,200,300]),
    ('fu__feat2__norm__norm', ['l1', 'l2']),
    ('kbest__k', [1,2,3,4,5]),
    ('svm__C', [1,10,100,1000]),
])

X, y = make_classification(n_samples=1000, n_features=1000)

n_folds = 5
n_jobs = 4
verbose = 1

start = time.time()
model = PipelineGridSearchCV(clone(pipe), cv_params, cv=n_folds, verbose=verbose, n_jobs=n_jobs,
                             mode='file', cachedir='file_cache', datasetname='make_class')
model.fit(X,y) # This will run much faster than ordinary GridSearchCV
elapsed_pipeline_grid_search = time.time() - start

start = time.time()
model = GridSearchCV(clone(pipe), cv_params, cv=n_folds, verbose=verbose, n_jobs=n_jobs)
model.fit(X,y)
elapsed_grid_search = time.time() - start

print("-----------------------------------")
print("Elapsed time for doing grid search:")
print("PipelineGridSearchCV: {} secs".format(elapsed_pipeline_grid_search))
print("GridSearchCV: {} secs".format(elapsed_grid_search))
print("Speedup: {:.2f}x".format(elapsed_grid_search / elapsed_pipeline_grid_search))

