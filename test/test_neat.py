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

"""Tests neat module"""

import unittest
import random as rng
import math as mt
import os
import neat.neat as nt

QUICK = False


class SimpleNeuralNetworkTestCase(unittest.TestCase):
    """Base class for NeuralNetwork testing"""
    def setUp(self):
        rng.seed(0)
        self.net = nt.NeuralNetwork(1, 1)


class NeuralNetworkTestCase(SimpleNeuralNetworkTestCase):
    """Tests NeuralNetwork class"""
    def test_put(self):
        """Tests whether network output is a list"""
        outs = self.net.put([1])
        self.assertTrue(isinstance(outs, list), 'Output of network is not a '
                                                'list')

    def test_create_empty(self):
        """Tests whether default constructor makes net empty"""
        res = self.net.put([1])
        self.assertAlmostEqual(res[0], 0, msg='Network is not empty.')

    def test_create_full(self):
        """Tests whether net can be created fully connected"""
        self.net = nt.NeuralNetwork.fully_connected(1, 1)
        res = self.net.put([1])
        self.assertNotAlmostEqual(res[0], 0, msg='Network is not connected.')
        res1 = self.net.put([0])
        self.assertNotAlmostEqual(res1[0], 0, msg='Bias not added.')
        self.assertNotAlmostEqual(res1[0], res[0], msg='The same result '
                                                       'regardless the input.')

    def test_copy(self):
        """Tests whether copying creates new instance with equal parameters"""
        net = self.net.copy()
        self.assertEquals(repr(net), repr(self.net), "Copy has different "
                                                     "parameters.")
        net.put([1])
        self.assertNotEquals(repr(net), repr(self.net), "Copy is the same "
                                                        "instance.")

    def test_is_similar(self):
        """Tests whether similarity measure works as intended"""
        net = self.net.copy()
        self.assertTrue(net.is_similar(self.net), "Network dissimilar to "
                                                  "copy.")
        net = nt.NeuralNetwork.fully_connected(1, 1)
        self.assertFalse(net.is_similar(self.net, d1=0.5), "Fully connected "
                                                           "similar to empty.")

    def test_complexity_measure(self):
        """Tests introduced complexity measure"""
        complexity1 = self.net.complexity()
        complexity2 = nt.NeuralNetwork.fully_connected(1, 1).complexity()
        complexity3 = nt.NeuralNetwork.fully_connected(5, 1).complexity()
        self.assertTrue(complexity1 < complexity2, "Fully connected network "
                        "less complex than empty.")
        self.assertTrue(complexity2 < complexity3, "Network with many "
                        "connections less complex than with less connections.")
        net = nt.NeuralNetwork.fully_connected(5, 1)
        # pylint: disable=protected-access
        edge = net._g.edges()[0]
        # pylint: disable=protected-access
        net._g[edge[0]][edge[1]]['weight'] = 1000
        complexity4 = net.complexity()
        self.assertTrue(complexity3 < complexity4, "Network with big weights "
                        "less complex than with small.")

    def test_save_and_restore(self):
        """Tests whether network after restore from file is the same"""
        self.net.put([2], auto_reset=False)
        filename = 'test_net.net'
        self.net.save(filename)
        loaded = nt.NeuralNetwork.load(filename)
        os.remove(filename)
        self.assertEqual(repr(self.net), repr(loaded), "Loaded network with "
                                                       "different parameters.")
        result1 = self.net.put([1], auto_reset=False)
        result2 = loaded.put([1], auto_reset=False)
        self.assertAlmostEqual(result1, result2, 5,
                               "Reloaded network returns different results.")


class SimpleNeatTestCase(unittest.TestCase):
    """Provides base for testing NEAT"""
    def setUp(self):
        rng.seed(0)
        self.neat = nt.Neat(2, 1)


class NeatTestCase(SimpleNeatTestCase):
    """Tests NEAT algorithm in general"""
    def test_save_and_restore(self):
        """Tests if restore from file returns identical object"""
        fname = 'test_neat.neat'
        self.neat.save(fname)
        loaded = nt.Neat.load(fname)
        os.remove(fname)
        self.assertEqual(repr(self.neat), repr(loaded),
                         "Loaded NEAT with different parameters.")
        rng.seed(0)
        best1, fit1 = self.neat.evolve(test_inputs=[1, 1],
                                       desired_outputs=[-1])
        rng.seed(0)
        best2, fit2 = loaded.evolve(test_inputs=[1, 1], desired_outputs=[-1])
        self.assertAlmostEqual(fit1, fit2, 5,
                               "Returned net with different fitness.")
        self.assertEqual(repr(best1), repr(best2),
                         "Returned different best net.")


@unittest.skipIf(QUICK, "Quick test - skipping XOR.")
class XorNeatTestCase(SimpleNeatTestCase):
    """Tests NEAT implementation with XOR"""
    def setUp(self):
        super(XorNeatTestCase, self).setUp()
        self.ins = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        self.outs = [[-1.0], [1.0], [1.0], [-1.0]]

    def test_learn_xor_by_goal(self):
        """Tests learning XOR through goal function"""
        self.neat.connection_add_mutation_probability = 0.8

        def _net_fitness(net):
            sum_of_squares = 0
            for idx in range(len(self.ins)):
                sum_of_squares += sum(
                    (e - o) ** 2 for e, o in zip(
                        self.outs[idx], net.put(self.ins[idx])))
            return -mt.sqrt(sum_of_squares)

        for _ in range(30):
            best, _ = self.neat.evolve(net_fitness=_net_fitness)

        for ins, outs in zip(self.ins, self.outs):
            pred = best.put(ins)
            self.assertTrue(pred[0] * outs[0] > 0, msg="Did not converge for "
                            "fitness function at %s giving %s instead of %s."
                            % (ins, pred, outs))

    def test_learn_xor_by_samples(self):
        """Tests learning XOR by samples"""
        self.neat.connection_add_mutation_probability = 0.9
        self.neat.complexity_penalty = 0.3

        for _ in range(30):
            for ins, outs in zip(self.ins, self.outs):
                self.neat.evolve(test_inputs=ins, desired_outputs=outs)

        def _net_fitness(net):
            sum_of_squares = 0
            for idx in range(len(self.ins)):
                sum_of_squares += sum((e - o) ** 2 for e, o in zip(
                    self.outs[idx], net.put(self.ins[idx])))
            return -mt.sqrt(sum_of_squares)

        best, _ = self.neat.evolve(net_fitness=_net_fitness)

        for ins, outs in zip(self.ins, self.outs):
            pred = best.put(ins)
            self.assertTrue(pred[0] * outs[0] > 0, msg="Did not converge for "
                            "individuals at %s giving %s instead of %s."
                            % (ins, pred, outs))


if __name__ == '__main__':
    unittest.main()
