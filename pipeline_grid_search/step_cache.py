
import os
import time
import warnings

import json
import numpy as np

class StepCache(object):
    def __init__(self, cachedir, datasetname=None):
        super(StepCache, self).__init__()
        self.cachedir = cachedir
        if datasetname is None:
            self.datasetname = self.cachedir.split('/')[-1]
        else:
            self.datasetname = datasetname

    def get_cache_filename(self, foldname, params_tail, stepname):
        """
        Returns the unique (up to hash collisions) filename
        of a transform path determined by params_tail.

        Keyword args:
        - foldname: Name for the cross-validation fold that
                    this cache refers to. Must include information
                    about if the fold is a train or test fold.
        - params_tail: List of (param_name: param_val) tuples of
                       params until (and including) the current step.
        """
        param_hash = hash(tuple(params_tail))
        filename = '{}/{}-{}-{}-{}.ftc'.format(self.cachedir, self.datasetname, foldname, param_hash, stepname)
        return filename

    def is_cached(self, foldname, params_tail, stepname):
        filename = self.get_cache_filename(foldname, params_tail, stepname)
        try:
            with open(filename, 'r') as f:
                d = json.load(f)
        except IOError:
            return False
        # Make extra check to avoid false positives due to hash collisions
        cached_params_tail = d.get('params_tail')
        if cached_params_tail is None:
            return False
        cached_params_tail = map(tuple, cached_params_tail)

        if cached_params_tail == params_tail:
            return True
        else:
            warnings.warn("Hash collision occured in StepCache file: {}\nDifference between cached_params_tail and params_tail is:\n{}".format(filename, set(tuple(cached_params_tail)) ^ set(tuple(params_tail))))
            return False

    def load_outdata(self, foldname, params_tail, stepname):
        filename = self.get_cache_filename(foldname, params_tail, stepname)
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
        return X

    def save_outdata(self, foldname, params_tail, stepname, fit_time, transform_time, X):
        filename = self.get_cache_filename(foldname, params_tail, stepname)
        outdata_filename = os.path.splitext(filename)[0] + '.npy'
        d = {
                'date': time.time(),
                'dataset': self.datasetname,
                'fold': foldname,
                'stepname': stepname,
                'params_tail': params_tail,
                'outdata_filename': outdata_filename,
                'fit_time': fit_time,
                'transform_time': transform_time,
            }
        with open(filename, 'w') as f:
            json.dump(d,f)
        with open(outdata_filename, 'w') as f:
            np.save(f, X)

