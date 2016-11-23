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

"""Tests whether caching works as intended"""

import unittest
import os
import random as rng
import sys
import caching.caching as caching

CACHE_PATH = os.path.join('.', 'cacher_test')


class CleanCacheTestCase(unittest.TestCase):
    """Checks whether cache is cleaned properly"""
    def setUp(self):
        @caching.cache_results(cache_path=CACHE_PATH)
        def _cache_something():
            pass

        _cache_something()

    def test_dir_removal(self):
        """Tests whether cache directory is removed"""
        self.assertTrue(os.path.exists(CACHE_PATH), 'Test failure - no '
                                                    'sample cache.')
        caching.clean_cache(CACHE_PATH)
        self.assertFalse(os.path.exists(CACHE_PATH), 'Did not remove cache.')


class CaptureTestCase(unittest.TestCase):
    """Tests various scenarios of using stdout capture"""
    def setUp(self):
        self.path = 'capture.test'
        self._stdout = sys.stdout
        sys.stdout = open(self.path, 'w')
        self.messages = ["test", "another one", "hell yeah!", str(range(10))]

    def tearDown(self):
        out_file = sys.stdout
        sys.stdout = self._stdout
        out_file.close()
        os.remove(self.path)

    def test_capturing_mode(self):
        """Tests whether whole stdout has been captured"""
        with caching.Capture() as capturing:
            for message in self.messages:
                print message
        self.assertEqual(len(capturing), len(self.messages), "Some messages "
                                                             "were dropped.")
        for message, captured in zip(self.messages, capturing):
            self.assertEqual(message, captured, "Expected message: "
                                                "%s\nGiven: %s" % (message,
                                                                   captured))

    def test_printout_to_stdout(self):
        """Tests whether messages have been displayed"""
        with caching.Capture() as _:
            for message in self.messages:
                print message
        sys.stdout.flush()
        with open(self.path, 'r') as out_file:
            lines = out_file.read().splitlines()
            self.assertEqual(len(lines), len(self.messages), "Some messages "
                                                             "were dropped.")
            for message, saved in zip(self.messages, lines):
                self.assertEqual(message, saved, "Expected message: "
                                                 "%s\nGiven: %s" % (message,
                                                                    saved))


class CacherTestCase(unittest.TestCase):
    """Tests various aspects of result caching"""
    def tearDown(self):
        caching.clean_cache(path=CACHE_PATH)

    def test_simple_caching(self):
        """Tests basic cache properties

        Checks whether cache is created, if the result from the cache is
        proper, if the function has been entered for the second time, etc.
        """
        entered_function = [False]

        @caching.cache_results(cache_path=CACHE_PATH)
        def _any_cached_function(arg, kwarg=None):
            entered_function[0] = True
            return "arg: %s, kwarg: %s" % (arg, kwarg)

        self.assertFalse(entered_function[0], "Test failed - wrong setup.")
        arg = 1
        first_res = _any_cached_function(arg)
        self.assertTrue(os.path.exists(CACHE_PATH), "Cache in unknown "
                                                    "location.")
        self.assertTrue(entered_function[0], "Did not enter the function in "
                                             "first call.")
        entered_function[0] = False
        second_res = _any_cached_function(arg)
        self.assertFalse(entered_function[0], "Entered function in second "
                                              "call.")
        self.assertEqual(first_res, second_res, "Return value changed.")
        third_res = _any_cached_function(arg, kwarg=None)
        self.assertEqual(first_res, third_res, "Return value changed on kwarg "
                                               "supply.")

    # pylint: disable=invalid-name
    # Keeps it descriptive
    def test_function_redefinition_robustness(self):
        """Tests whether cacher is robust after function re-definition

        Checks if cache is working properly when address of a function in
        memory changes.
        """

        entered_function = [False]

        @caching.cache_results(cache_path=CACHE_PATH)
        def _any_cached_function(arg, kwarg=None):
            entered_function[0] = True
            return "arg: %s, kwarg: %s" % (arg, kwarg)

        self.assertFalse(entered_function[0], "Test failed - wrong setup.")
        arg = 1
        first_res = _any_cached_function(arg)
        self.assertTrue(os.path.exists(CACHE_PATH), "Cache in unknown "
                                                    "location.")
        self.assertTrue(entered_function[0], "Did not enter the function in "
                                             "first call.")

        @caching.cache_results(cache_path=CACHE_PATH)
        # pylint: disable=function-redefined
        # This behaviour is under test
        def _any_cached_function(arg, kwarg=None):
            entered_function[0] = True
            return "arg: %s, kwarg: %s" % (arg, kwarg)

        entered_function[0] = False
        second_res = _any_cached_function(arg)
        self.assertFalse(entered_function[0], "Entered function in second "
                                              "call.")
        self.assertEqual(first_res, second_res, "Return value changed.")
        third_res = _any_cached_function(arg, kwarg=None)
        self.assertEqual(first_res, third_res, "Return value changed on kwarg "
                                               "supply.")

    # pylint: disable=too-many-locals
    # Those variables are necessary to check results
    def test_rng_preservation(self):
        """Tests whether caching is rng-sensible"""
        @caching.cache_results(cache_path=CACHE_PATH)
        def _any_cached_function(arg, kwarg=None):
            return arg, kwarg, rng.random()

        arg = 1
        rng.seed(0)
        a1, k1, r1 = _any_cached_function(arg)
        state1 = rng.getstate()
        a2, k2, r2 = _any_cached_function(arg)
        state2 = rng.getstate()
        self.assertEqual(a1, a2, "Return arg changed.")
        self.assertEqual(k1, k2, "Returned kwarg changed.")
        self.assertNotEqual(r1, r2, "Returned the same random.")

        rng.seed(0)
        rep_a1, rep_k1, rep_r1 = _any_cached_function(arg)
        rep_state1 = rng.getstate()
        rep_a2, rep_k2, rep_r2 = _any_cached_function(arg)
        rep_state2 = rng.getstate()
        self.assertEqual(a1, rep_a1, "Return arg changed after rng reset.")
        self.assertEqual(k1, rep_k1, "Return kwarg changed after rng reset.")
        self.assertEqual(r1, rep_r1, "Returned different cached random after "
                                     "rng reset.")
        self.assertEqual(state1, rep_state1, "State of rng after first call "
                                             "after reset is different.")
        self.assertEqual(rep_a1, rep_a2, "Return arg2 changed after rng "
                                         "reset.")
        self.assertEqual(rep_k1, rep_k2, "Return kwarg2 changed after rng "
                                         "reset.")
        self.assertEqual(r2, rep_r2, "Returned different cached random2 "
                                     "after rng reset.")
        self.assertEqual(state2, rep_state2, "State of rng after second call "
                                             "after rng reset is different.")

    def test_stdout_preservation(self):
        """Tests whether stdout has been captured and printed"""
        @caching.cache_results(cache_path=CACHE_PATH)
        def _function_with_printout(arg, kwarg=None):
            print "arg: %s, kwarg: %s" % (arg, kwarg)
            return arg, kwarg

        stdout = sys.stdout
        try:
            sys.stdout = open('log1.test', 'w')
            _function_with_printout(1)
            sys.stdout.close()
            sys.stdout = open('log2.test', 'w')
            _function_with_printout(1)
            sys.stdout.close()
        finally:
            sys.stdout = stdout
        with open('log1.test', 'r') as f1, open('log2.test', 'r') as f2:
            self.assertEqual(f1.read(), f2.read(), "Reports differ.")
        os.remove('log1.test')
        os.remove('log2.test')


if __name__ == '__main__':
    unittest.main()
