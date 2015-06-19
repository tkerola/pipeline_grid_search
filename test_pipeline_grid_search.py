"""
Testing for pipeline_grid_search module.
"""

from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from nose.tools import assert_equal

from pipeline_grid_search import PipelineGridSearchCV

# Globals for counting estimator calls
n_transform_calls = 0
n_fit_calls = 0

# http://stackoverflow.com/a/27005560/4963543
def make_init_body(classname,parameters):
    # Calling super does not work for some reason,
    # but it does not matter in this case, since
    # BaseEstimator, TransformerMixin and ClassifierMixin have an empty __init__ function.
    #body = "        super({}, self).__init__()".format(classname)
    body = "        pass"
    body += ''.join('\n        self.{}={}'.format(key,key) for key,_ in parameters)
    func_str = "    def __init__(self{}):\n{}".format(''.join(', {}={}'.format(key,val) for key,val in parameters), body)
    return func_str

def create_mock_estimator(classname,parameters,is_classifier=False):
    # parameters is a list of (key,val) pairs.

    init_body = make_init_body(classname,parameters)

    main_body = """
    def fit(self, X, y=None):
        global n_fit_calls
        n_fit_calls += 1
        return self
    """
    if is_classifier:
        bases = "(BaseEstimator, TransformerMixin, ClassifierMixin)"
        main_body += """
    def predict(self, X):
        return np.arange(X.shape[0])
        """
    else:
        bases = "(BaseEstimator, TransformerMixin)"
        main_body += """
    def transform(self, X):
        global n_transform_calls
        n_transform_calls += 1
        return X
        """
    body = "class {}{}:\n{}\n{}".format(classname,bases,init_body,main_body)

    print(body)
    exec(body)

    newclassobj = locals()[classname]()
    return newclassobj

def create_mock_classifier(classname,parameters):
    return create_mock_estimator(classname,parameters,is_classifier=True)

def calc_n_ideal_fit_calls(parts, cv_params, n_folds):
    def nfits(nparams):
        # calcs the number of optimal calls to fit when following DFS order
        # (the number of nodes in the pipeline tree minus one)
        s = 1
        for c in reversed(nparams):
            s = 1+c*s if c>1 else s+1
        return s-1

    pipe_length = len(parts)
    nparams = []
    for p in parts:
        param_count = 1
        for (name,vals) in cv_params:
            est_name,_ = name.split("__",1)
            if est_name == p.__class__.__name__:
                param_count *= len(vals)
        nparams.append(param_count)
        
    n_ideal_fit_calls = nfits(nparams)
    n_ideal_fit_calls *= n_folds        # We repeat the above number of fit calls for each fold 
    n_ideal_fit_calls += pipe_length    # plus the fits for fitting on the whole X last

    return n_ideal_fit_calls

def test_pipeline_grid_search():
    # The that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_estimator("f0",[]),
        create_mock_estimator("f1", [("p1",0),("p2",2)]),
        create_mock_estimator("f2",[]),
        create_mock_estimator("f3",[("c",0),("d",0)]),
        create_mock_estimator("f4",[]),
        create_mock_estimator("f5",[]),
        create_mock_classifier("f6",[("c",0)]),
        ]

    pipe = Pipeline([ (p.__class__.__name__, p) for p in parts ])

    cv_params = [
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
        ('f6__c', [10,20,30,40]),
    ]

    X, y = make_classification(n_samples=100, n_features=20)

    n_folds = 5
    n_jobs = 1

    # mock.MagicMock cannot be used since GridSearchCV resets each estimator using
    # clone() before each call to fit.
    # So, let's use global variables instead that we increment in our mock
    # estimators.
    global n_transform_calls, n_fit_calls

    # Start PipelineGridSearchCV test here
    n_transform_calls = 0
    n_fit_calls = 0
    model = PipelineGridSearchCV(pipe, dict(cv_params), cv=n_folds, verbose=1, n_jobs=n_jobs)
    model.fit(X,y)
    print("Counts (PipelineGridSearchCV)")
    print("n_transform_calls:",n_transform_calls)
    print("n_fit_calls:",n_fit_calls)

    n_ideal_fit_calls = calc_n_ideal_fit_calls(parts,cv_params,n_folds)
    # Make sure that PipelineGridSearchCV only called fit the optimal number of times.
    assert_equal(n_fit_calls, n_ideal_fit_calls)

    # Start GridSearchCV test here
    n_transform_calls = 0
    n_fit_calls = 0
    model = GridSearchCV(pipe, dict(cv_params), cv=n_folds, verbose=1, n_jobs=n_jobs)
    model.fit(X,y)
    print("Counts (GridSearchCV)")
    print("n_transform_calls:",n_transform_calls)
    print("n_fit_calls:",n_fit_calls)

    n_naive_fit_calls = np.prod(map(lambda x: len(x[1]), cv_params)) * len(parts) * n_folds + len(parts)
    assert_equal(n_fit_calls, n_naive_fit_calls)

#test_pipeline_grid_search()
