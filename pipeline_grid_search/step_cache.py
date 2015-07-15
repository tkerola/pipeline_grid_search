
import os
import time
import warnings

import json
import numpy as np

class StepCache(object):
    def __init__(self, cachedir, datasetname=None, use_memory_cache=True):
        super(StepCache, self).__init__()
        self.cachedir = cachedir
        if datasetname is None:
            self.datasetname = self.cachedir.split('/')[-1]
        else:
            self.datasetname = datasetname
        self.seen_params_tails = {}
        self.seen_outdata = {}
        self.use_memory_cache = use_memory_cache

    def get_cache_filename(self, foldname, active_params, canonical_stepname):
        """
        Returns the unique (up to hash collisions) filename
        of a transform path determined by active_params.

        Keyword args:
        - foldname: Name for the cross-validation fold that
                    this cache refers to. Must include information
                    about if the fold is a train or test fold.
        - active_params: List of (param_name: param_val) tuples of
                       params until (and including) the current step.
        """
        param_hash = hash(tuple(active_params))
        filename = '{}/{}-{}-{}-{}.ftc'.format(self.cachedir, self.datasetname, foldname, param_hash, canonical_stepname)
        return filename

    def is_cached(self, foldname, active_params, canonical_stepname):
        filename = self.get_cache_filename(foldname, active_params, canonical_stepname)
        # For speed, first check memory
        cached_params_tail = self.seen_params_tails.get(filename) if self.use_memory_cache else None
        if cached_params_tail is None:
            try:
                with open(filename, 'r') as f:
                    d = json.load(f)
            except IOError:
                return False
            # Make extra check to avoid false positives due to hash collisions
            cached_params_tail = d.get('active_params')
            if cached_params_tail is None:
                return False
            cached_params_tail = map(tuple, cached_params_tail)
            if self.use_memory_cache:
                self.seen_params_tails[filename] = cached_params_tail

        if cached_params_tail == active_params:
            return True
        else:
            warnings.warn("Hash collision occured in StepCache file: {}\nDifference between cached_params_tail and active_params is:\n{}".format(filename, set(tuple(cached_params_tail)) ^ set(tuple(active_params))))
            return False

    def load_outdata(self, foldname, active_params, canonical_stepname):
        filename = self.get_cache_filename(foldname, active_params, canonical_stepname)
        X = self.seen_outdata.get(filename) if self.use_memory_cache else None
        if X is None:
            try:
                with open(filename, 'r') as f:
                    d = json.load(f)
            except IOError as ioe:
                print("Error: Cache file does not exist.")
                raise ioe
            outdata_filename = d.get('outdata_filename')
            if outdata_filename is None:
                raise ValueError('outdata_filename is not specified in cache file {}'.format(filename))
            X = np.load(outdata_filename)
            if self.use_memory_cache:
                self.seen_outdata[filename] = X # NOTE: Add pruning of dict once we store too many elements? (limit memory usage)
        return X

    def save_outdata(self, canonical_stepname, X, foldname, active_params, fit_time, transform_time, fit_transform_time):
        filename = self.get_cache_filename(foldname, active_params, canonical_stepname)
        outdata_filename = os.path.splitext(filename)[0] + '.npy'
        d = {
                'date': time.time(),
                'dataset': self.datasetname,
                'fold': foldname,
                'canonical_stepname': canonical_stepname,
                'active_params': active_params,
                'outdata_filename': outdata_filename,
                'fit_time': fit_time,
                'transform_time': transform_time,
            }
        with open(filename, 'w') as f:
            json.dump(d,f)
        with open(outdata_filename, 'w') as f:
            np.save(f, X)

