
PipelineGridSearchCV
====================
An optimized version of GridSearchCV and Pipeline that
avoids unnecessary calls to fit()/transform() by doing grid search in
depth first search (DFS) order.

Motivation
==========
Running GridSearchCV on a Pipeline classifier will cause
a full pipeline computation for each set of parameters,
despite some parameters being the same as before,
causing unnecessary calls to fit/transform.

Doing 5-fold CV using GridSearchCV on a Pipeline will
cause `prod(param_counts)*pipe_length*5+pipe_length` calls to `fit()`,
where `prod(param_counts)` is the product of the number of grid search
parameters for each part of the pipeline, and `pipe_length` is the
length of the pipeline.

PipelineGridSearchCV, on the other hand, requires only the optimal
`(nverts-1)*5+pipe_length` calls to `fit()`,
where `nverts` is the number of vertices in the pipeline parameter call
tree.

The idea is that the doing grid search on a pipeline estimator forms
a directed tree of computation, where each grid search parameter causes
the tree to split into a new subtree.
 
As an example, consider a the following Pipeline and grid search parameters:
```python
pipe = Pipeline([
    ('pca', PCA()), 
    ('norm', Normalizer()), 
    ('svm', SVC())
    ])

cv_params = {
    'pca__n_components': [100,200],
    'norm__norm': ['l2'],
    'svm__C': [1,10,100],
    }

model = PipelineGridSearchCV(pipe, cv_params)
```

Doing grid search on this pipeline will result in the following grid search tree.
![Pipeline grid search tree](https://cloud.githubusercontent.com/assets/3026734/8267030/675c638c-178a-11e5-8efe-bf0d52ddd8b9.png)

We can see by looking at the tree that calling `model.fit()` from the root of the tree for each set of paramters is unnecessary;
it is sufficient to call it in depth first search (DFS) order if the parameters are updated in that order as well.

Usage
=====
```
$ python examples/example.py
-----------------------------------
Elapsed time for doing grid search:
PipelineGridSearchCV: 12.456952095 secs
GridSearchCV: 38.9708809853 secs
```

Dependencies
============
Tested with:
```
numpy 1.9.2
scikit-learn 0.16.1
nosetets 1.3.4 (for testing)
```

Tests
=====
Make sure that everything is working correctly by running the test suite:
```
$ make test
```

License
=======
BSD 3-Clause (see LICENSE)
