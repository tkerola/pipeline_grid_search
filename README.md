
PipelineGridSearchCV
====================
An optimized usage case of scikit-learn's `GridSearchCV` and `Pipeline` that
avoids unnecessary calls to `fit()/transform()` by doing grid search in
depth first search (DFS) order.

Motivation
==========
Running GridSearchCV on a Pipeline classifier will cause
a full pipeline computation for each set of parameters,
despite some parameters being the same as before,
causing unnecessary calls to fit/transform.

Doing 5-fold CV using `GridSearchCV` on a Pipeline will
cause `prod(param_counts)*pipe_length*5+pipe_length` calls to `fit()`,
where `prod(param_counts)` is the product of the number of grid search
parameters for each part of the pipeline, and `pipe_length` is the
length of the pipeline.

`PipelineGridSearchCV`, on the other hand, requires only the optimal
`(nverts-1)*5+pipe_length` calls to `fit()`,
where `nverts` is the number of vertices in the pipeline parameter call
tree.

The idea is that the doing grid search on a pipeline estimator forms
a directed tree of computation, where each grid search parameter causes
the tree to split into a new subtree.
 
As an example, consider the following pipeline and grid search parameters:
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

We have `2*1*3 = 6` parameter combinations to evaluate using grid search.
Doing grid search on this pipeline will result in the following grid search tree.
![Pipeline grid search tree](https://cloud.githubusercontent.com/assets/3026734/8267030/675c638c-178a-11e5-8efe-bf0d52ddd8b9.png)

Each pass from the root to a leaf corresponds to one specific selection of parameters for
the estimators in the pipeline.
We can see that calling `model.fit()` from the root of the tree for each set of parameters is unnecessary;
it is sufficient to call it from nodes in depth first search (DFS) order if the parameters are updated in that order as well.
For example, if we have previously called `fit()` from the root with the parameters `{'pca__n_components': 100, 'norm__norm': 'l2', 'svm__C': 100}`
and then want to try the parameters `{'pca__n_components': 100, 'norm__norm': 'l2', 'svm__C': 10}`, then we only need to restart the pipeline
computation from the SVM step; only the `svm__C` parameter has changed.

While `GridSearchCV` will call `fit()` from the root each time, `PipelineGridSearchCV` keeps track of the order of the searched
parameters, and calls fit only from the nodes necessary in order to evaluate a new parameter choice, avoiding unnecessary repeated computation.

Below is a comparison of the number of calls to `fit`, comparing the optimized and naive approach to grid search.
The pipe grid search structure for the above example corresponds to pipe param. count [2, 1, 3].

| Pipe param. count | #calls PipelineGridSearchCV (A) | #calls GridSearchCV (B) | A/B |
| --------------------- | ------------------------------- | ----------------------- | --- |
| [2, 1, 3]             | 10                              | 18                      | 0.555 |
| [10, 20, 30, 20]      | 126210                          | 480000                  | 0.263 |
| [1, 1, 1, 10, 1, 20, 20] |  4223                        | 28000                   | 0.151 |

We can see that PipelineGridSearchCV gives better performance the deeper and wider the pipeline grid search gets (smaller A/B ratio).

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
python 2.7.3
numpy 1.9.2
scikit-learn 0.16.1
nosetests 1.3.4 (for testing)
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
