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

from sklearn.base import is_classifier, clone, BaseEstimator
from sklearn.cross_validation import check_cv, _fit_and_score, _safe_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline, FeatureUnion 
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

class _CacheGridSearchCVPipeline(Pipeline):

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
        self.grid_search_mode = False
        self.verbose = verbose

        self.cache = cache

    def store_cv_data(self, cv_params, cv_foldname):
        self.cv_params_steps = _get_params_steps(cv_params)
        self.cv_foldname = cv_foldname

        for name,transform in self.steps:
            if isinstance(transform, _CacheGridSearchCVPipeline):
                transform.store_cv_data(cv_params, cv_foldname)
            elif isinstance(transform, FeatureUnion):
                for fu_name,fu_transform in transform.transformer_list:
                    # NOTE: This should be made into a recursive check function.
                    if isinstance(fu_transform, _CacheGridSearchCVPipeline):
                        fu_transform.store_cv_data(cv_params, cv_foldname)

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

            fit_time = 0.

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
        

class _DFSGridSearchCVPipeline(Pipeline):

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

        self.grid_search_mode = False
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
        if not self.grid_search_mode:
            return X, 0

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
            # Visualize from which step in the pipeline we (re)start the transform.
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

    def store_cv_data(self, cv_params, cv_foldname):
        if self.cv_params_steps:  # Keep track of the previous cv params
            self.cv_params_steps_prev = self.cv_params_steps
        self.cv_params_steps = _get_params_steps(cv_params)
        self.cv_foldname = cv_foldname

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
            if hasattr(transform, "fit_transform"):
                Xt = transform.fit_transform(Xt, y, **fit_params_steps[name])
            else:
                Xt = transform.fit(Xt, y, **fit_params_steps[name]) \
                              .transform(Xt)
            if self.grid_search_mode:
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

            if self.grid_search_mode:
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


def _fit_and_score_pipeline_grid_fold(
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

    grid_estimator.grid_search_mode = True

    if verbose > 1:
        print("_fit_and_score_pipeline_grid_fold")

    out = []
    for parameters in parameter_iterable:

        try:
            # Bookkeeping of the previously calculated
            # steps does not work if we clone
            # the pipeline as clone(grid_estimator),
            # as the lists fit_transform_history etc. are
            # reset each iteration.
            # Instead, just clone the subestimators.
            grid_estimator.store_cv_data(parameters, "f{}".format(fold_index))

            # Note that we use a custom clone method here
            # that skips cloning _DFSGridSearchCVPipeline,
            # and only clones the pipeline steps.
            # This is needed in order to keep track of previous
            # calculations during grid search, which allows
            # us to avoid repeated calculations of the same
            # pipeline step.
            grid_estimator.clone_steps()
            grid_estimator.set_params(**parameters)  # Set params of subestimators

            ret = _fit_and_score(grid_estimator, X, y, scorer,
                                 train, test, verbose, parameters, fit_params,
                                 False, return_parameters, error_score)
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

            raise exception
        finally:
            out.append(ret)

    grid_estimator.grid_search_mode = False

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
            grid_estimator = _DFSGridSearchCVPipeline(
                clone(self.estimator), self.param_grid, self.verbose)
        elif self.mode == 'file':
            stepcache = StepCache(self.cachedir, self.datasetname)
            grid_estimator = _CacheGridSearchCVPipeline(clone(self.estimator), stepcache, self.verbose)
            if not os.path.isdir(self.cachedir):
                if os.path.isfile(self.cachedir):
                    raise ValueError("Specified cachedir is a file. Cannot overwrite {}".format(self.cachedir))
                os.makedirs(self.cachedir)
            else:
                pattern_regex = r'{}-.*\.(ftc|npy)'.format(re.escape(self.datasetname))
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

        def add_sub_dfs_pipelines(est, params):
            try:
                params_steps = _get_params_steps(params)
            except ValueError: # This happens when we reach an estimator without any subestimators.
                return

            if not isinstance(est, (Pipeline,FeatureUnion)):
                return
            steps = est.steps if isinstance(est, Pipeline) else est.transformer_list

            for i,(step_name,step) in enumerate(steps):

                if isinstance(step, Pipeline):
                    if self.mode == 'dfs':
                        pipe = _DFSGridSearchCVPipeline(step, params_steps[step_name], self.verbose)
                    else:
                        pipe = _CacheGridSearchCVPipeline(step, stepcache, self.verbose)
                    steps[i] = (step_name,pipe)
                    add_sub_dfs_pipelines(pipe, params_steps[step_name])
                else:
                    add_sub_dfs_pipelines(step, params_steps[step_name])

        # Search for any embedded Pipelines and replace them with a _*GridSearchCVPipeline
        add_sub_dfs_pipelines(grid_estimator, self.param_grid)

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

        def extract_sub_stepnames(stepname):
            step = self.pipeline.named_steps[stepname]
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
                pipe_stepnames = extract_sub_stepnames(pipe_stepnames[i1])
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

