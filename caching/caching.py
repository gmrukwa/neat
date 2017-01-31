# Copyright 2016 Grzegorz Mrukwa
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Provides decorator for long-lasting function result caching purposes"""

import functools
import os.path
import shutil
import random as rng
import numpy.random as nprng
import pickle
from cStringIO import StringIO
import sys

__all__ = ['cache_results', 'clean_cache', 'Capture']


class Capture(list):
    """Class for stdout capturing purpose"""

    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        # No sense to define __init__
        self._stdout = sys.stdout
        # pylint: disable=attribute-defined-outside-init
        # No sense to define __init__
        self._stringio = StringIO()
        sys.stdout = self
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout

    def write(self, text):
        """Writes to stdout with capture"""
        self._stringio.write(text)
        self._stdout.write(text)

    def flush(self):
        """Performs flush on stdout"""
        self._stdout.flush()


def _get_fun_help(fun):
    stdout = sys.stdout
    sys.stdout = StringIO()
    help(fun)
    helpstr = sys.stdout
    sys.stdout = stdout
    # pylint: disable=no-member
    # getvalue is member of StringIO
    return helpstr.getvalue()


def cache_results(cache_path=os.path.join('.', 'cache'), function_tag='f',
                  with_rng=True, debug=False):
    """Creates decorator for caching results of a function calls on disc

    :param cache_path: path, under which cache will be created
    :param function_tag: if defined, uses to separate function caching domains
    :param with_rng: if true, checks rng state and uses its value to save
    results with randomization
    :param debug: if true some debug info is printed
    :return: decorator for caching
    """
    def _log(msg):
        if debug:
            print msg
    _log("Adjusting decorator...\n")

    def _adjusted_cacher(fun):
        _log("Decorating a function...\n")

        @functools.wraps(fun)
        def _cacher(*args, **kwargs):
            _log("Using a decorated function...\n")

            def _hs(arg):
                if callable(arg):
                    return str(hex(hash(_get_fun_help(arg))))
                if isinstance(arg, (tuple, list)):
                    if len(arg) > 1:
                        return _hs(_hs(arg[0]) + _hs(arg[1:]))
                    if len(arg) > 0:
                        return _hs(arg[0])
                    return str(hex(hash(arg)))
                elif isinstance(arg, dict):
                    acc = ""
                    for key in arg:
                        acc += _hs(arg[key])
                    return _hs(acc)
                return str(hex(hash(str(arg))))

            _log("Hashed a function.\n")
            args_hash = _hs(args)
            _log("Hashed *args.\n")
            kwargs_hash = _hs(kwargs)
            _log("Hashed **kwargs.\n")
            if with_rng:
                rng_hash = _hs(rng.getstate()) + _hs(nprng.get_state())
            else:
                rng_hash = ""
            _log("Hashed rng.\n")
            path = os.path.join(cache_path, function_tag, args_hash)
            if with_rng:
                path = os.path.join(path, rng_hash)
            _log("Searching for path %s...\n" % path)
            if not os.path.exists(path):
                _log("Making non-existent directory for cache.\n")
                os.makedirs(path)
            cachefile_path = os.path.join(path, kwargs_hash) + '.cache'
            randomseed_path = os.path.join(path, kwargs_hash) + '.rng'
            printout_path = os.path.join(path, kwargs_hash) + '.out'
            _log("Cache file: %s\n" % cachefile_path)
            if os.path.isfile(cachefile_path):
                _log("Found existing results. Loading... \n")
                if with_rng:
                    with open(randomseed_path, 'r') as cache_file:
                        normal, numpied = pickle.load(cache_file)
                        rng.setstate(normal)
                        nprng.set_state(numpied)
                with open(printout_path, 'r') as cache_file:
                    print cache_file.read()
                with open(cachefile_path, 'r') as cache_file:
                    return pickle.load(cache_file)
            else:
                _log("No existing results found. Calculating...\n")
                with Capture() as captured_output:
                    res = fun(*args, **kwargs)
                with open(cachefile_path, 'w') as cache_file:
                    pickle.dump(res, cache_file)
                with open(printout_path, 'w') as cache_file:
                    cache_file.writelines(captured_output)
                if with_rng:
                    with open(randomseed_path, 'w') as cache_file:
                        dump = (rng.getstate(), nprng.get_state())
                        pickle.dump(dump, cache_file)
                return res
        return _cacher
    return _adjusted_cacher


def clean_cache(path=os.path.join('.', 'cache')):
    """Removes whole cache in directory

    :param path: path to cache directory
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
