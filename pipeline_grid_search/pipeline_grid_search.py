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

import itertools
from collections import Sized, defaultdict

import numpy as np

from sklearn.base import is_classifier, clone
from sklearn.cross_validation import check_cv, _fit_and_score, _safe_split
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import _num_samples, indexable

# Note: These counters do not work correctly with clone
# and Parallel, as they get reset by each thread and clone.
# n_fit_transform_calls = 0
# n_score_transform_calls = 0


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

    def store_cv_params(self, **cv_params):
        if self.cv_params_steps:  # Keep track of the previous cv params
            self.cv_params_steps_prev = self.cv_params_steps
        self.cv_params_steps = _get_params_steps(cv_params)

    def clone_steps(self):
        """
        Clones the steps of this pipeline,
        starting from the appropriate starting step,
        effectively resetting them, while
        leaving the current instance of this pipeline
        intact.
        """
        # start_step depends on store_cv_params,
        # so it should be called after store_cv_params
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
        train, test, verbose,
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
            grid_estimator.store_cv_params(**parameters)

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
        except:
            # In case fit threw an exception, store the result, with NaNs.
            ret = []
            X_test, y_test = _safe_split(grid_estimator, X, y, test, train)
            ret.extend([np.float("NaN"), _num_samples(X_test), np.float("NaN")])
            if return_parameters:
                ret.append(parameters)
        finally:
            out.append(ret)

    grid_estimator.grid_search_mode = False

    return out


class PipelineGridSearchCV(GridSearchCV):

    def __init__(self, *args, **kwargs):
        super(PipelineGridSearchCV, self).__init__(*args, **kwargs)

        if not isinstance(self.estimator, Pipeline):
            raise ValueError(
                "PipelineGridSearchCV only accepts Pipeline as its estimator.")

    def fit(self, X, y=None):
        return self._fit(X, y,
                         _DFSParameterGrid(self.param_grid, self.estimator))

    def _fit(self, X, y, parameter_iterable):
        """Actual fitting,  performing the search over parameters."""

        # Make the supplied Pipeline _DFSGridSearchCVPipeline
        # for doing the grid_search
        grid_estimator = _DFSGridSearchCVPipeline(
            self.estimator, self.param_grid, self.verbose)

        def add_sub_dfs_pipelines(params, steps):
            params_steps = _get_params_steps(params)
            for i,(step_name,step) in enumerate(steps):
                # Replace any Pipeline with _DFSGridSearchCVPipeline
                if isinstance(step,Pipeline):
                    pipe = _DFSGridSearchCVPipeline(step, params_steps[step_name], self.verbose)
                    steps[i] = (step_name,pipe)
                    add_sub_dfs_pipelines(params_steps[step_name], pipe.steps)

        # Search for any embedded Pipelines and replace them with _DFSGridSearchCVPipeline
        add_sub_dfs_pipelines(self.param_grid, grid_estimator.steps)

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
                train, test, self.verbose, parameter_iterable,
                self.fit_params, return_parameters=True,
                error_score=self.error_score)
            for train, test in cv)

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

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            # Note that we use self.estimator here instead
            # of grid_estimator in order to keep self.best_estimator_
            # as a normal Pipeline. _DFSGridSearchCVPipeline is
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

    def __init__(self, param_grid, pipeline):
        super(_DFSParameterGrid, self).__init__(param_grid)
        self.pipeline = pipeline

    def __iter__(self):
        # Returns parameters in DFS order according to
        # the pipeline structure.

        for p in self.param_grid:
            # Sort for reproducibility.
            sorted_items = sorted(p.items())

            items = []
            # Ensure items are in the same order as steps
            for step_name, _ in self.pipeline.steps:
                # Assume estimator__param syntax of params.
                for pname, pval in sorted_items:
                    step, param = pname.split('__', 1)
                    if step == step_name:
                        items.append((pname, pval))

            # Now items appear in DFS order,
            # so itertools.product below will
            # return them in DFS order as well.

            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in itertools.product(*values):
                    params = dict(zip(keys, v))
                    yield params

