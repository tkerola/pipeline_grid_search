import time

from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from pipeline_grid_search import PipelineGridSearchCV

pipe = Pipeline([
    ("pca", PCA()),
    ("svm", SVC()),
    ])

cv_params = dict([
    ('pca__n_components', [100,200,300]),
    ('svm__C', [1,10,100,1000]),
])

X, y = make_classification(n_samples=1000, n_features=1000)

n_folds = 5
n_jobs = 4
verbose = 1

start = time.time()
model = PipelineGridSearchCV(pipe, cv_params, cv=n_folds, verbose=verbose, n_jobs=n_jobs)
model.fit(X,y) # This will run much faster than ordinary GridSearchCV
elapsed_pipeline_grid_search = time.time() - start

start = time.time()
model = GridSearchCV(pipe, cv_params, cv=n_folds, verbose=verbose, n_jobs=n_jobs)
model.fit(X,y)
elapsed_grid_search = time.time() - start

print("-----------------------------------")
print("Elapsed time for doing grid search:")
print("PipelineGridSearchCV: {} secs".format(elapsed_pipeline_grid_search))
print("GridSearchCV: {} secs".format(elapsed_grid_search))
