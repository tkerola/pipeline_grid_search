"""
Testing for pipeline_grid_search module.
"""

from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

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

def nfits(nparams):
    # calcs the number of optimal calls to fit when following DFS order
    # (the number of nodes in the pipeline tree minus one)
    s = 1
    for c in reversed(nparams):
        s = 1+c*s if c>1 else s+1
    return s-1

def calc_n_ideal_fit_calls(parts, cv_params, n_folds):
    pipe_length = len(parts)
    nparams = []
    for p in parts:
        param_count = 1
        for (name,vals) in cv_params:
            est_name,_ = name.split("__",1)
            if est_name == p.__class__.__name__:
                param_count *= len(vals)
        nparams.append(param_count)
    print(nparams)
        
    n_ideal_calls = nfits(nparams)
    n_ideal_calls *= n_folds        # We repeat the above number of fit calls for each fold 
    n_ideal_calls += pipe_length    # plus the fits for fitting on the whole X last

    return n_ideal_calls

def calc_n_ideal_transform_calls(parts, cv_params, n_folds):
    pipe_length = len(parts)
    nparams = []
    for p in parts[:-1]: # Do not include the last part of the pipeline; it is a classifier (without transform)
        param_count = 1
        for (name,vals) in cv_params:
            est_name,_ = name.split("__",1)
            if est_name == p.__class__.__name__:
                param_count *= len(vals)
        nparams.append(param_count)
        
    n_ideal_calls = nfits(nparams)
    n_ideal_calls *= n_folds*2        # We repeat the above number of fit calls for each fold (and for both the train and development set)
    n_ideal_calls += pipe_length-1    # plus the fits for fitting on the whole X last (minus the classifier at the end)

    return n_ideal_calls

def test_pipeline_grid_search1():
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

    cv_params = [
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
        ('f6__c', [10,20,30,40]),
    ]

    perform_pipeline_case(parts, cv_params)

def test_pipeline_grid_search2():
    # The that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_estimator("f0",[]),
        create_mock_estimator("f1", [("p1",0),("p2",2)]),
        create_mock_estimator("f2",[]),
        create_mock_estimator("f3",[("c",0),("d",0)]),
        create_mock_estimator("f4",[]),
        create_mock_estimator("f5",[]),
        create_mock_estimator("f40",[]),
        create_mock_estimator("f50",[]),
        create_mock_estimator("f41",[]),
        create_mock_estimator("f51",[]),
        create_mock_estimator("f42",[]),
        create_mock_estimator("f52",[]),
        create_mock_classifier("f6",[("c",0)]),
        ]

    cv_params = [
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
        ('f6__c', [10,20,30,40]),
    ]

    perform_pipeline_case(parts, cv_params)

def test_pipeline_grid_search3():
    # The that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_classifier("f1", [("p1",0)]),
        ]

    cv_params = [
        ('f1__p1', [10,20]),
    ]

    perform_pipeline_case(parts, cv_params)

def test_pipeline_grid_search4():
    # The that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_classifier("f1", []),
        ]

    cv_params = [
    ]

    perform_pipeline_case(parts, cv_params)

def test_pipeline_grid_search5():
    # The that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_estimator("f0",[]),
        create_mock_estimator("f1", [("p1",0),("p2",2)]),
        create_mock_estimator("f2",[]),
        create_mock_estimator("f3",[("c",0),("d",0)]),
        create_mock_estimator("f4",[]),
        create_mock_estimator("f5",[]),
        create_mock_estimator("f6",[]),
        create_mock_estimator("f7",[]),
        create_mock_estimator("f8",[]),
        create_mock_estimator("f9",[]),
        create_mock_estimator("f10",[]),
        create_mock_classifier("f11",[]),
        ]

    cv_params = [
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
    ]

    perform_pipeline_case(parts, cv_params)

def test_pipeline_grid_search6():
    # Test that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        create_mock_estimator("f0",[]),
        create_mock_estimator("f1", [("p1",0),("p2",2)]),
        create_mock_estimator("f2",[]),
        create_mock_estimator("f3",[("c",0),("d",0)]),
        create_mock_estimator("f4",[]),
        create_mock_estimator("f5",[]),
        SVC() 
        ]

    cv_params = [
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
        ('SVC__C', [1.,10.,100.,1000.]),
    ]

    # Set assert_n_calls_equal to False, as we need to implement our custom counting of function calls in order to measure the call tests.
    perform_pipeline_case(parts, cv_params, assert_n_calls_equal=False)

def test_pipeline_grid_search7():
    # Test that the number of estimator calls is less than the ones for regular GridSearchCV
    parts = [
        PCA(),
        Normalizer(),
        SVC()
        ]

    cv_params = [
        ('PCA__n_components', [3,5,7]),
        ('Normalizer__norm', ['l2']),
        ('SVC__C', [1.,10.,100.,1000.]),
    ]

    perform_pipeline_case(parts, cv_params, assert_n_calls_equal=False)

def test_pipeline_grid_search8():
    # Test using a FeatureUnion with embedded Pipelines.
    parts = [
        create_mock_estimator("f0",[]),
        FeatureUnion([
            ('feat1', Pipeline([
                ('f11', create_mock_estimator("f11", [("p1",0),("p2",2)])),
                ])),
            ('feat2', Pipeline([
                ('f12', create_mock_estimator("f12", [("a",0)])),
                ])),
            ]),
        create_mock_estimator("f1", [("p1",0),("p2",2)]),
        create_mock_estimator("f2",[]),
        create_mock_estimator("f3",[("c",0),("d",0)]),
        create_mock_estimator("f4",[]),
        create_mock_estimator("f5",[]),
        create_mock_classifier("f11",[]),
        ]

    cv_params = [
        ('FeatureUnion__feat1__f11__p1', [10,20]),
        ('FeatureUnion__feat2__f12__a', [10,20,30]),
        ('f1__p1', [10,20]),
        ('f3__c', [10,20,30]),
        ('f3__d', [10,20,30,40]),
    ]

    # Set assert_n_calls_equal to False, as we need to implement our custom counting of function calls in order to measure the call tests.
    perform_pipeline_case(parts, cv_params, assert_n_calls_equal=False)
    # TODO: Update assert_n_calls_equal logic to work correctly with pipelines embedded in FeatureUnions.

def perform_pipeline_case(parts, cv_params, assert_n_calls_equal=True):
    # tests a particular pipe and cv_params combination

    pipe = Pipeline([ (p.__class__.__name__, p) for p in parts ])
    print(pipe)

    X, y = make_classification(n_samples=100, n_features=20)

    n_folds = 5
    n_jobs = 1
    verbose = 0

    # mock.MagicMock cannot be used since GridSearchCV resets each estimator using
    # clone() before each call to fit.
    # So, let's use global variables instead that we increment in our mock
    # estimators.
    global n_transform_calls, n_fit_calls

    # Start PipelineGridSearchCV test here
    n_transform_calls = 0
    n_fit_calls = 0
    model = PipelineGridSearchCV(pipe, dict(cv_params), cv=n_folds, verbose=verbose, n_jobs=n_jobs)
    model.fit(X,y)
    print("model.best_estimator_: {}".format(model.best_estimator_))
    print("Counts (PipelineGridSearchCV)")
    print("n_fit_calls:",n_fit_calls)
    print("n_transform_calls:",n_transform_calls)

    n_ideal_fit_calls = calc_n_ideal_fit_calls(parts,cv_params,n_folds)
    n_ideal_transform_calls = calc_n_ideal_transform_calls(parts,cv_params,n_folds)
    if assert_n_calls_equal:
        # Make sure that PipelineGridSearchCV only called fit the optimal number of times.
        assert_equal(n_fit_calls, n_ideal_fit_calls)
        assert_equal(n_transform_calls, n_ideal_transform_calls)

    # Start GridSearchCV test here
    n_transform_calls = 0
    n_fit_calls = 0
    model_naive = GridSearchCV(pipe, dict(cv_params), cv=n_folds, verbose=verbose, n_jobs=n_jobs)
    model_naive.fit(X,y)
    print("Counts (GridSearchCV)")
    print("n_fit_calls:",n_fit_calls)
    print("n_transform_calls:",n_transform_calls)

    n_param_combs = np.prod(map(lambda x: len(x[1]), cv_params))
    n_naive_fit_calls = n_param_combs * len(parts) * n_folds + len(parts)
    n_naive_transform_calls = n_param_combs * (len(parts)-1) * n_folds * 2 + (len(parts)-1) # The 2 is for running on both the train and dev. set
    if assert_n_calls_equal:
        assert_equal(n_fit_calls, n_naive_fit_calls)
        assert_equal(n_transform_calls, n_naive_transform_calls)

    # Make sure that PipelineGridSearchCV and GridSearchCV return the same result.
    assert_equal(model_naive.best_score_, model.best_score_)
    assert_equal(model_naive.best_params_, model.best_params_)

