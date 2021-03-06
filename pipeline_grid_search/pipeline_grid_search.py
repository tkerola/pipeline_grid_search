"""
Provides PipelineGridSearchCV, as class for
doing efficient grid search in a Pipeline estimator
while avoiding unnecessary repeated calls to fit and score.

Updated for sklearn 0.16.1

License: BSD 3-Clause (see LICENSE)
This file contains partly rewritten source code from scikit-learn's pipeline.py and grid_search.py.
These are also under a BSD 3-Clause license (see https://github.com/scikit-learn/scikit-learn/blob/master/COPYING).
"""

from __future__ import print_function
from __future__ import division

import time
import itertools
import operator
import os
import re
from collections import Sized, defaultdict

import numpy as np
from scipy import sparse

from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.cross_validation import check_cv, _safe_split, _score
from sklearn.externals.joblib import Parallel, delayed, logger
from sklearn.externals import six
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline, FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one
from sklearn.utils.validation import _num_samples, indexable

from step_cache import StepCache

def _get_params_steps(params):
    """
    Maps d['estimator__param'] style keys to
    a nested d['estimator']['param'] style
    dictionary.
    """
    params_steps = defaultdict(dict)
    for pname, pval in six.iteritems(params):
        step, param = pname.split('__', 1)
        params_steps[step][param] = pval
    return params_steps

class _CVDataMixin(object):

    def store_cv_data(self, cv_params, cv_foldname):
        """
        Stores cross-validation data that the estimator
        should be aware of before calling fit/transform/score.
        """
        if hasattr(self, 'cv_params_steps'):  # Keep track of the previous cv params
            self.cv_params_steps_prev = self.cv_params_steps
        self.cv_params_steps = _get_params_steps(cv_params)
        self.cv_foldname = cv_foldname

    def set_foldname(self, foldname):
        self.cv_foldname = foldname

class _CacheGridSearchCVPipeline(Pipeline, _CVDataMixin):

    """
    This pipeline keeps a cache or previous
    computations in order to avoid repeated
    computation during grid search.

    Note that this estimator is not meant to
    be used outside this module, as it breaks
    some common estimator assumptions in order
    to avoid repeated computation.
    """

    def __init__(self, steps, cache, verbose=0):
        if isinstance(steps, Pipeline):
            super(_CacheGridSearchCVPipeline, self).__init__(steps.steps)
        else:
            super(_CacheGridSearchCVPipeline, self).__init__(steps)
        self.verbose = verbose

        self.cache = cache

    def format_step_params(self, name):
        return map(lambda (a,b): ('{}__{}'.format(name, a), b), sorted(self.cv_params_steps[name].items()))

    def _get_start_state(self, X, foldname):
        params_tail = []
        start_step = -1

        Xt = X

        # Find out how deep into the pipeline we have
        # existing cached values.
        for step_index,(name, transform) in enumerate(self.steps[:-1]):

            step_params = self.format_step_params(name)
            params_tail.extend(step_params)

            if not self.cache.is_cached(foldname, params_tail, name):
                start_step = step_index
                params_tail = params_tail[:-len(step_params)]
                if step_index == 0:
                    Xt = X
                else:
                    # Load cache of previous step
                    Xt = self.cache.load_outdata(foldname, params_tail, self.steps[step_index-1][0])
                break
        
        return Xt, start_step, params_tail

    def clone_steps(self):
        """
        Clones the steps of this pipeline, while
        leaving the current instance of this pipeline
        intact.
        """
        for i in xrange(len(self.steps)):
            subest_name, subest = self.steps[i]
            if isinstance(subest, _CacheGridSearchCVPipeline):
                subest.clone_steps()
            elif isinstance(subest, FeatureUnion):
                for j in xrange(len(subest.transformer_list)):
                    fu_subest_name, fu_subest = subest.transformer_list[j]
                    if isinstance(fu_subest, _CacheGridSearchCVPipeline):
                        fu_subest.clone_steps()
                    else:
                        fu_subest = clone(fu_subest)
                        subest.transformer_list[j] = (fu_subest_name, fu_subest)
            else:
                subest = clone(subest)
                self.steps[i] = (subest_name, subest)
                self.named_steps[subest_name] = subest

    def _pre_transform(self, X, y=None, **fit_params):
        # This method is only called from fit.
        # assume that this function is only called
        # on the train fold during CV
        foldname = self.cv_foldname + '_train'

        fit_params_steps = _get_params_steps(fit_params)
        Xt, start_step, params_tail = self._get_start_state(X, foldname)

        # Start computation from the non-cached step
        for name, transform in self.steps[start_step:-1]:

            step_params = self.format_step_params(name)
            params_tail.extend(step_params)

            if isinstance(transform, _CVDataMixin):
                transform.store_cv_data(self.cv_params_steps[name], self.cv_foldname)

            if hasattr(transform, "fit_transform"):
                fit_time = time.time()
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
                fit_time = time.time() - fit_time
                transform_time = None
            else:
                fit_time = time.time()
                transform.fit(Xt, y, **fit_params_steps[name])
                fit_time = time.time() - fit_time

                transform_time = time.time()
                Xt = transform.transform(Xt)
                transform_time = time.time() - transform_time

            self.cache.save_outdata(foldname, params_tail, name, fit_time, transform_time, Xt)

        return Xt, fit_params_steps[self.steps[-1][0]]

    def score(self, X, y=None):
        # assume that this function is only called
        # on the test fold during CV
        foldname = self.cv_foldname + '_test'

        Xt, start_step, params_tail = self._get_start_state(X, foldname)

        # Start computation from the non-cached step
        for name, transform in self.steps[start_step:-1]:

            step_params = self.format_step_params(name)
            params_tail.extend(step_params)

            fit_time = None

            transform_time = time.time()
            Xt = transform.transform(Xt)
            transform_time = time.time() - transform_time

            self.cache.save_outdata(foldname, params_tail, name, fit_time, transform_time, Xt)

        return self.steps[-1][-1].score(Xt, y)

    def get_params(self, deep=True):
        out = super(Pipeline, self).get_params(deep=False)
        if not deep:
            return out
        else:
            out.update(self.named_steps.copy())
            for name, step in six.iteritems(self.named_steps):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
        

class _DFSGridSearchCVPipeline(Pipeline, _CVDataMixin):
    """
    This pipeline assumes that when doing
    grid search, the parameters are being
    supplied in DFS order.

    Note that this estimator is not meant to
    be used outside this module, as it breaks
    some common estimator assumptions in order
    to avoid repeated computation.
    """

    def __init__(self, steps, param_grid, verbose=0):
        if isinstance(steps, Pipeline):
            super(_DFSGridSearchCVPipeline, self).__init__(steps.steps)
        else:
            super(_DFSGridSearchCVPipeline, self).__init__(steps)
        self.param_grid = param_grid
        self.verbose = verbose

        self.cv_params_counts = {}
        param_grid_steps = _get_params_steps(param_grid)
        # Store the maximum of the number of params to grid search
        # over for each step.
        for name, step in self.steps:
            step_param_counts = map(lambda x: len(x),
                                    param_grid_steps[name].values())
            self.cv_params_counts[name] = (max(step_param_counts)
                                           if step_param_counts else 0)

        # Keep track of previous transforms in the pipeline
        self.fit_transform_history = []
        self.score_transform_history = []
        # Keep track of the previous cv params of each step
        self.cv_params_steps = []
        self.cv_params_steps_prev = []

    def _get_start_state(self, X=None, fit_score_mode='fit'):
        """
        Returns the appropriate starting location
        in the pipeline computation flow by
        comparing current and previous CV parameters
        and assuming that the parameters appear
        in DFS order to the fit and score functions
        when doing grid search.

        Returns:
        - Xt: The feature at step start_step (if X is not None).
        - start_step: The starting step.
        """

        start_step = 0
        if self.cv_params_steps_prev:
            prev_i_with_params = len(self.steps) - 1
            for i, (step_name, _) in reversed(list(enumerate(self.steps))):
                cur_params = self.cv_params_steps[step_name]
                prev_params = self.cv_params_steps_prev[step_name]

                # Make sure that we do not start from a subestimator
                # that has no or just one cv_param to set.
                # However, if we are at the first step (i==0),
                # then we should always go to the cur_params == prev_params
                # statement below in order to avoid restarting from the start
                # of the chain even though the first estimator does not have
                # any CV parameters set to explore.
                if self.cv_params_counts[step_name] < 2 and i>0:
                    continue

                if cur_params == prev_params:
                    start_step = prev_i_with_params
                    break
                prev_i_with_params = i

        if self.verbose > 1 and fit_score_mode == 'fit' and not X is None:
            # Visualize from which step in the pipeline we (re)start
            # the transform.
            print("[{}] ".format(start_step+1)+"-"*(start_step)+">")

        if X is None:
            return start_step

        if fit_score_mode == 'fit':
            transform_history = self.fit_transform_history
        else:
            transform_history = self.score_transform_history

        if start_step == 0:
            # Reset history
            Xt = X
            del transform_history[:]
            transform_history.append(Xt)
        else:
            # Clear future and get result from prev step
            del transform_history[start_step+1:]
            Xt = transform_history[start_step]

        return Xt, start_step

    def clone_steps(self):
        """
        Clones the steps of this pipeline,
        starting from the appropriate starting step,
        effectively resetting them, while
        leaving the current instance of this pipeline
        intact.
        """
        # start_step depends on store_cv_data,
        # so it should be called after store_cv_data
        start_step = self._get_start_state()
        for i in xrange(start_step, len(self.steps)):
            subest_name, subest = self.steps[i]
            subest = clone(subest)
            self.steps[i] = (subest_name, subest)
            self.named_steps[subest_name] = subest

    def _pre_transform(self, X, y=None, **fit_params):
        # This method is only called from fit.

        # Save each step of previously computed values
        # of Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
        fit_params_steps = _get_params_steps(fit_params)
        Xt, start_step = self._get_start_state(X, 'fit')

        # Restart transform from step i (it has new params)
        for name, transform in self.steps[start_step:-1]:
            if isinstance(transform, _CVDataMixin):
                transform.store_cv_data(self.cv_params_steps[name], self.cv_foldname)

            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]) \
                              .transform(Xt)
            self.fit_transform_history.append(Xt)

            # global n_fit_transform_calls
            # n_fit_transform_calls += 1
            # print("*** n_fit_transform_calls = {} ***".format(
            #        n_fit_transform_calls))

        return Xt, fit_params_steps[self.steps[-1][0]]

    def score(self, X, y=None):
        Xt, start_step = self._get_start_state(X, 'score')
        for name, transform in self.steps[start_step:-1]:
            Xt = transform.transform(Xt)

            self.score_transform_history.append(Xt)

            # global n_score_transform_calls
            # n_score_transform_calls += 1
            # print("*** n_score_transform_calls = {} ***".format(
            #       n_score_transform_calls))

        return self.steps[-1][-1].score(Xt, y)

    def get_params(self, deep=True):
        out = super(Pipeline, self).get_params(deep=False)
        if not deep:
            return out
        else:
            out.update(self.named_steps.copy())
            for name, step in six.iteritems(self.named_steps):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

def _canonical_step_params(canonical_prefix, params):
    return map(lambda (param_name, param_val): ((canonical_prefix+param_name), param_val), sorted(params.items()))

def _create_subest_canonial_prefix(canonical_prefix, stepname):
    return canonical_prefix + stepname + "__"

def _get_start_state(cache, steps, X, foldname, canonical_prefix, cv_params_steps):
    active_params = []
    start_step = -1

    # Find out how deep into the pipeline we have
    # existing cached values.
    for step_index,(stepname, transform) in enumerate(steps):

        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, stepname)
        canonical_step_params = _canonical_step_params(subest_canonial_prefix, cv_params_steps[stepname])
        active_params.extend(canonical_step_params)
        canonical_stepname = canonical_prefix + stepname

        if not cache.is_cached(foldname, active_params, canonical_stepname):
            start_step = step_index
            active_params = active_params[:-len(canonical_step_params)]
            if step_index == 0:
                Xt = X
            else:
                # Load cache of previous step
                prev_stepname = steps[step_index-1][0]
                prev_canonial_stepname = canonical_prefix + prev_stepname
                Xt = cache.load_outdata(foldname, active_params, prev_canonial_stepname)
            break
    
    return Xt, start_step, active_params

def _do_cached_fit_transform(cache, canonical_estname, est, canonical_prefix, cv_params, X, y, scorer, foldname, active_params, mode, **fit_params):
    # entering this function means that we have requested to
    # recompute the value of this estimator.

    canonical_step_params = _canonical_step_params(canonical_prefix, cv_params)
    estname = canonical_estname.split("__")[-1]

    elapsed = time.time()

    if isinstance(est, Pipeline):
        fit_params_steps = _get_params_steps(fit_params)
        cv_params_steps = _get_params_steps(cv_params)

        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, estname)
        local_active_params = list(active_params) # Make a copy in order to avoid modidying in place
        Xt = _do_cached_pipeline_fit_transform_or_score(cache, est.steps, subest_canonial_prefix, cv_params_steps[estname], X, y, scorer, foldname, local_active_params, mode, **fit_params_steps[estname])
    elif isinstance(est, FeatureUnion):
        fit_params_steps = _get_params_steps(fit_params)
        cv_params_steps = _get_params_steps(cv_params)

        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, estname)
        local_active_params = list(active_params) # Make a copy in order to avoid modidying in place
        Xt = _do_cached_feature_union_fit_transform(cache, est.transformer_list, subest_canonial_prefix, cv_params_steps[estname], X, y, scorer, foldname, local_active_params, mode, **fit_params_steps[estname])
    else:
        # Do the actual fit/transform

        if mode.startswith('fit'):
            if hasattr(est, "fit_transform"):
                Xt = est.fit_transform(X, y, **fit_params)
            else:
                Xt = est.fit(X, y, **fit_params) \
                              .est(Xt)
        else:
            Xt = est.transform(X)

    # Activate estimator params only after any eventual Pipeline or FeatureUnion subcase has been
    # handled in order to not mess with cache logic.
    active_params.extend(canonical_step_params)
    elapsed = time.time() - elapsed
    if mode.startswith('fit'):
        _post_transform(cache, canonical_estname, Xt, foldname, active_params, fit_transform_time=elapsed)
    else:
        _post_transform(cache, canonical_estname, Xt, foldname, active_params, transform_time=elapsed)

    return Xt

def _do_cached_feature_union_fit_transform(cache, steps, canonical_prefix, cv_params, X, y, scorer, foldname, active_params, mode, **fit_params):
    # mode = 'fit', 'fit_transform' or 'transform'

    fit_params_steps = _get_params_steps(fit_params)
    cv_params_steps = _get_params_steps(cv_params)

    Xts = []
    for fu_name, fu_trans in steps:
        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, fu_name)
        canonical_step_params = _canonical_step_params(subest_canonial_prefix, cv_params_steps[fu_name])
        local_active_params = active_params + canonical_step_params
        canonical_stepname = canonical_prefix + fu_name
        if cache.is_cached(foldname, local_active_params, canonical_stepname):
            # Load cache of step
            Xti = cache.load_outdata(foldname, local_active_params, canonical_stepname)
        else:
            elapsed = time.time()
            Xti = _do_cached_fit_transform(cache, canonical_stepname, fu_trans, subest_canonial_prefix, cv_params_steps[fu_name], X, y, scorer, foldname, local_active_params, mode, **fit_params_steps[fu_name])

            elapsed = time.time() - elapsed
            if mode.startswith('fit'):
                _post_transform(cache, canonical_stepname, Xti, foldname, local_active_params, fit_transform_time=elapsed)
            else:
                _post_transform(cache, canonical_stepname, Xti, foldname, local_active_params, transform_time=elapsed)

        Xts.append(Xti)
    Xt = np.hstack(Xts)

    return Xt

def _do_cached_pipeline_fit_transform_or_score(cache, steps, canonical_prefix, cv_params, X, y, scorer, foldname, active_params=None, mode=None, **fit_params):
    # mode = 'fit', 'fit_transform', 'fit_score', 'transform' or 'score'
    if mode is None:
        raise ValueError("must specify mode ('fit', 'fit_transform', 'fit_score', 'transform' or 'score')")
    if active_params is None:
        active_params = []

    # For subestimators:
    # 'fit':            NOT VALID
    # 'fit_transform':  VALID
    # 'fit_score':      NOT VALID
    # 'transform':      VALID
    # 'score':          NOT VALID
    if mode == 'fit':
        # Any submerged pipelines must return transformed data.
        subest_mode = 'fit_transform' 
    elif 'score' in mode:
        # for subestimators, 'score' -> 'transform'
        # scoring is only valid for the topmost pipeline
        # Any submerged pipelines must return transformed features, not scores.
        subest_mode = mode.replace('score', 'transform')
    else:
        subest_mode = mode

    fit_params_steps = _get_params_steps(fit_params)
    cv_params_steps = _get_params_steps(cv_params)

    # Start computing the head of the pipeline.
    Xt, start_index, local_active_params = _get_start_state(cache, steps, X, foldname, canonical_prefix, cv_params_steps)
    print("[{}] restart from step {}".format(foldname,steps[start_index][0]))
    print("local_active_params:",local_active_params)
    print("active_params:",active_params)
    active_params.extend(local_active_params)
    for stepname, step in steps[start_index:-1]:
        print("doing step {}".format(stepname))
        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, stepname)
        subest_canonial_stepname = canonical_prefix + stepname
        Xt = _do_cached_fit_transform(cache, subest_canonial_stepname, step, subest_canonial_prefix, cv_params_steps[stepname], Xt, y, scorer, foldname, active_params, subest_mode, **fit_params_steps[stepname])

    # The last step of a pipeline is special,
    # so let's treat it separately at the end.
    stepname, step = steps[-1]
    if mode == 'fit':
        print("doing end of pipe fit, step {}".format(stepname))
        step.fit(Xt, y)
    elif mode.endswith('transform'):
        if mode.startswith('fit'):
            step.fit(Xt, y)
            print("doing end of pipe fit[transform], step {}".format(stepname))
        subest_canonial_prefix = _create_subest_canonial_prefix(canonical_prefix, stepname)
        subest_canonial_stepname = canonical_prefix + stepname
        print("doing end of pipe transform, step {}".format(stepname))
        Xt = _do_cached_fit_transform(cache, subest_canonial_stepname, step, subest_canonial_prefix, cv_params_steps[stepname], Xt, y, scorer, foldname, active_params, subest_mode, **fit_params_steps[stepname])
        print("[end of pipe] active_params:",active_params)
        return Xt
    elif mode.endswith('score'):
        if mode.startswith('fit'):
            step.fit(Xt, y)
            print("doing end of pipe fit[score], step {}".format(stepname))
        print("doing end of pipe score, step {}".format(stepname))
        the_score = _score(step, Xt, y, scorer)
        return the_score
    else:
        raise ValueError("Invalid mode: {}".format(mode))

def _post_transform(cache, canonical_stepname, Xt, foldname, active_params,
                    fit_time=None, transform_time=None, fit_transform_time=None):
    cache.save_outdata(canonical_stepname, Xt, foldname, active_params, fit_time, transform_time, fit_transform_time)

def _fit_and_score_one_pipe(
        cache,
        pipe, X, y, scorer,
        fold_index, train, test, verbose,
        cv_params, fit_params,
        return_train_score=False,
        return_parameters=False, error_score='raise'):

    if verbose > 1:
        if cv_params is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in cv_params.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(X, v, train))
                      for k, v in fit_params.items()])

    pipe = clone(pipe)

    if cv_params is not None:
        pipe.set_params(**cv_params)

    start_time = time.time()

    X_train, y_train = _safe_split(pipe, X, y, train)
    X_test, y_test = _safe_split(pipe, X, y, test, train)

    train_foldname = "f{}_train".format(fold_index)
    test_foldname = "f{}_test".format(fold_index)

    canonical_prefix = ''

    # fit_score
    try:
        if return_train_score:
            train_score = _do_cached_pipeline_fit_transform_or_score(cache, pipe.steps, canonical_prefix, cv_params, X_train, y_train, scorer, train_foldname, mode='fit_score', **fit_params)
        else:
            _do_cached_pipeline_fit_transform_or_score(cache, pipe.steps, canonical_prefix, cv_params, X_train, y_train, scorer, train_foldname, mode='fit', **fit_params)

    except Exception as e:
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)"
                             )

    else:
        test_score = _do_cached_pipeline_fit_transform_or_score(cache, pipe.steps, canonical_prefix, cv_params, X_test, y_test, scorer, test_foldname, mode='score')

    scoring_time = time.time() - start_time

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(X_test), scoring_time])
    if return_parameters:
        ret.append(cv_params)
    return ret

def _fit_and_score_pipeline_grid_fold(
        mode, cachedir, datasetname,
        grid_estimator, X, y, scorer,
        fold_index, train, test, verbose,
        parameter_iterable, fit_params,
        return_parameters=False, error_score='raise'):
    """
    Fits and scores all available parameters while keeping
    the train-test fold fixed in order to make use of
    _DFSGridSearchCVPipeline.
    """

    # Note that we have removed the return_train_score parameter
    # above, as this would break the assumption _DFSGridSearchCVPipeline
    # that grid_estimator.fit is only called on X_train and
    # grid_estimator.score is only called on X_test.

    if verbose > 1:
        print("_fit_and_score_pipeline_grid_fold")

    if mode == 'file':
        cache = StepCache(cachedir, datasetname)
    else:
        cache = None

    out = []
    for parameters in parameter_iterable:

        try:
            ret = _fit_and_score_one_pipe(
                    cache,
                    grid_estimator, X, y, scorer,
                    fold_index, train, test, verbose,
                    parameters, fit_params,
                    False,
                    return_parameters, error_score)
        except Exception as exception:
            # In case fit threw an exception, store the result, with NaNs.
            ret = []
            X_test, y_test = _safe_split(grid_estimator, X, y, test, train)
            ret.extend([np.float("NaN"), _num_samples(X_test), np.float("NaN")])
            if return_parameters:
                ret.append(parameters)

            # Print exception output for debugging/logging.
            print("Caught exception during fit. Traceback below:")
            print("=============================================")
            import traceback
            traceback.print_exc()

            raise
        finally:
            out.append(ret)

    return out


class PipelineGridSearchCV(GridSearchCV):

    def __init__(self, estimator, param_grid, mode='dfs', cachedir=None, datasetname=None,
                 *args, **kwargs):
        super(PipelineGridSearchCV, self).__init__(estimator, param_grid, *args, **kwargs)

        if not isinstance(self.estimator, Pipeline):
            raise ValueError(
                "PipelineGridSearchCV only accepts Pipeline as its estimator.")

        self.mode = mode
        if mode == 'file':
            if cachedir is None:
                raise ValueError("Must specify cachedir when mode is 'file'.")
        self.cachedir = cachedir
        self.datasetname = datasetname

    def fit(self, X, y=None):
        return self._fit(X, y,
                         _DFSParameterGrid(self.param_grid, self.estimator))

    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        if self.mode == 'dfs':
            # Make the supplied Pipeline a _DFSGridSearchCVPipeline
            # for doing the grid_search
            #grid_estimator = _DFSGridSearchCVPipeline(
            #    clone(self.estimator), self.param_grid, self.verbose)
            grid_estimator = clone(self.estimator)
        elif self.mode == 'file':
            #grid_estimator = _CacheGridSearchCVPipeline(clone(self.estimator), stepcache, self.verbose)
            grid_estimator = clone(self.estimator)

            # Cache dir handling
            if not os.path.isdir(self.cachedir):
                if os.path.isfile(self.cachedir):
                    raise ValueError("Specified cachedir is a file. Cannot overwrite {}".format(self.cachedir))
                os.makedirs(self.cachedir)
            else:
                pattern_regex = r'{}+.*\.(ftc|npy)'.format(re.escape(self.datasetname))
                pattern = re.compile(pattern_regex)
                nfound = 0
                for f in os.listdir(self.cachedir):
                    if pattern.search(f):
                        p = os.path.join(self.cachedir, f)
                        os.remove(os.path.join(self.cachedir, f))
                        nfound += 1
                if self.verbose > 0:
                    print("Removed {} existing files matching '{}' in cache dir ({}).".format(nfound, pattern_regex, self.cachedir))
        else:
            raise ValueError("Invalid mode specified.")

        cv = self.cv
        self.scorer_ = check_scoring(grid_estimator, scoring=self.scoring)

        n_samples = _num_samples(X)
        X, y = indexable(X, y)

        if y is not None:
            if len(y) != n_samples:
                raise ValueError('Target variable (y) has a different number '
                                 'of samples (%i) than data (X: %i samples)'
                                 % (len(y), n_samples))
        cv = check_cv(cv, X, y, classifier=is_classifier(grid_estimator))

        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates,
                                         n_candidates * len(cv)))

        pre_dispatch = self.pre_dispatch

        # Note: Compared to GridSearchCV._fit,
        # the order of cv and parameter_iterable have been
        # switched below as to fit with the same fold data
        # throughout all the possible parameters in the pipeline
        # before switching fold, allowing us to take advantage
        # of the pipeline structure to avoid repeated computation.
        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(
            delayed(_fit_and_score_pipeline_grid_fold)(
                self.mode, self.cachedir, self.datasetname,
                clone(grid_estimator), X, y, self.scorer_,
                fold_index, train, test, self.verbose, parameter_iterable,
                self.fit_params, return_parameters=True,
                error_score=self.error_score)
            for fold_index, (train, test) in enumerate(cv))

        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = 0
        for fold in out:
            n_fits += len(fold)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        # Since we have switched we order of cv and parameter_iterable
        # above, the indexing below has also been updated
        # accordingly.
        fold_offset = n_fits // n_folds
        for parameter_set_index in xrange(0, fold_offset):
            n_test_samples = 0
            score = 0
            all_scores = []
            for fold in out:
                this_score, this_n_test_samples, _, parameters = fold[
                    parameter_set_index]
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            # TODO: shall we also store the test_fold_sizes?
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        print("PipelineGridSearchCV: grid_estimator after fit: {}".format(grid_estimator))

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            # Note that we use self.estimator here instead
            # of grid_estimator in order to keep self.best_estimator_
            # as a normal Pipeline. _*GridSearchCVPipeline is
            # only useful when doing grid search.
            best_estimator = clone(self.estimator).set_params(
                **best.parameters)
            if y is not None:
                best_estimator.fit(X, y, **self.fit_params)
            else:
                best_estimator.fit(X, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self


class _DFSParameterGrid(ParameterGrid):

    """
    ParameterGrid that returns items
    in depth first search (DFS) order.
    """

    def __init__(self, param_grid, pipeline, enumerate_params=False):
        super(_DFSParameterGrid, self).__init__(param_grid)
        self.pipeline = pipeline
        self.enumerate_params = enumerate_params

    def __iter__(self):
        # Returns parameters in DFS order according to
        # the pipeline structure.

        def extract_sub_stepnames(step):
            if isinstance(step, Pipeline):
               return map(operator.itemgetter(0), step.steps)
            elif isinstance(step, FeatureUnion):
               return map(operator.itemgetter(0), step.transformer_list)
            return []

        def cmp_func((pname1, vals1), (pname2, vals2)):
            parts1 = pname1.split('__')
            parts2 = pname2.split('__')
            nested_stepnames1 = parts1[:-1]
            nested_stepnames2 = parts2[:-1]
            param1 = parts1[-1]
            param2 = parts2[-1]
            pipe_stepnames = map(operator.itemgetter(0), self.pipeline.steps)
            step = self.pipeline
            for n1,n2 in itertools.izip_longest(nested_stepnames1, nested_stepnames2):
                if n1 is None:
                    return 1
                elif n2 is None:
                    return -1
                i1 = pipe_stepnames.index(n1)
                i2 = pipe_stepnames.index(n2)
                if i1 < i2:
                    return -1
                elif i1 > i2:
                    return 1
                if isinstance(step, Pipeline):
                    step = step.named_steps[pipe_stepnames[i1]]
                elif isinstance(step, FeatureUnion):
                    step = step.transformer_list[i1][1]
                else:
                    raise ValueError("Unsupported subestimator: {}".format(step))
                pipe_stepnames = extract_sub_stepnames(step)
                if len(pipe_stepnames) == 0:
                    break

            # step is equal, so let param name and values determine order
            return cmp((param1,vals2), (param2,vals2))

        for p in self.param_grid:
            # Sort for reproducibility (according to DFS order).
            sorted_items = sorted(p.items(), cmp=cmp_func)

            # Now items appear in DFS order,
            # so itertools.product below will
            # return them in DFS order as well.

            if not sorted_items:
                yield {}
            else:
                keys, values = zip(*sorted_items)
                for v in itertools.product(*values):
                    params = dict(zip(keys, v))
                    yield params

def _pipeline_grid_search_fit_one_transformer(transformer, X, y, parameters, foldname):
    if isinstance(transformer, (_DFSGridSearchCVPipeline, _CacheGridSearchCVPipeline)):
        transformer.store_cv_data(parameters, foldname)
        transformer.clone_steps()
        transformer.set_params(**parameters)

    return _fit_one_transformer(transformer, X, y)

def _pipeline_grid_search_fit_transform_one(transformer, name, X, y, transformer_weights,
                        parameters, foldname,
                       **fit_params):
    if isinstance(transformer, (_DFSGridSearchCVPipeline, _CacheGridSearchCVPipeline)):
        transformer.store_cv_data(parameters, foldname)
        transformer.clone_steps()
        transformer.set_params(**parameters)

    return _fit_transform_one(transformer, name, X, y, transformer_weights, **fit_params)

def _pipeline_grid_search_transform_one(transformer, name, X, transformer_weights, parameters, foldname):
    if isinstance(transformer, (_DFSGridSearchCVPipeline, _CacheGridSearchCVPipeline)):
        transformer.store_cv_data(parameters, foldname)
        transformer.clone_steps()
        transformer.set_params(**parameters)

    return _transform_one(transformer, name, X, transformer_weights)

class _GridSearchCVFeatureUnion(FeatureUnion, _CVDataMixin):
    """
    A FeatureUnion that calls handles embedded _*GridSearchCVPipeline
    estimators correctly.

    Note: This class should never be used directly. Only by PipelineGridSearchCV.
    """
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        super(_GridSearchCVFeatureUnion, self).__init__(transformer_list, n_jobs, transformer_weights)

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data, used to fit transformers.
        """
        transformers = Parallel(n_jobs=self.n_jobs)(
            delayed(_pipeline_grid_search_fit_one_transformer)(trans, X, y, self.cv_params_steps[name], self.cv_foldname)
            for name, trans in self.transformer_list)
        self._update_transformer_list(transformers)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers using X, transform the data and concatenate
        results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_pipeline_grid_search_fit_transform_one)(trans, name, X, y,
                                        self.transformer_weights,
                                        self.cv_params_steps[name], self.cv_foldname,
                                        **fit_params)
            for name, trans in self.transformer_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_pipeline_grid_search_transform_one)(trans, name, X, self.transformer_weights, self.cv_params_steps[name], self.cv_foldname)
            for name, trans in self.transformer_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

