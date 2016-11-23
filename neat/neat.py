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

"""Provides implementation of NEAT algorithm.

The module implements an algorithm of Neuro Evolution of Augmenting
Topologies (NEAT) and simple artificial neural network implementation.

NeuralNetwork class provided allows to estimate the output values for
specified inputs and to check similarity of two networks.

Neat class main purpose is genetic/evolutionary training of neural networks -
their weights and topologies.  Interpretation of: Stanley, K. O., Miikkulainen,
R. (2002) Efficient Reinforcement Learning through Evolving Neural Network
Topologies. In Proc. GECCO-2002 'The Genetic and Evolutionary Computation'.
"""

import random as rng
from math import tanh, sqrt, exp, ceil
from warnings import warn
import param
from tqdm import tqdm
import ast
import networkx as nx
import numpy as np

__all__ = ['NeuralNetwork', 'Neat']


def _unzip(collection):
    return zip(*collection)


def _weighted_sample(collection, weights):
    assert len(collection) == len(
        weights), "Collection and weights lengths differ."
    for w in weights:
        assert w >= 0, "Cannot use negative weights."
    r = rng.uniform(0.0, sum(weights))
    for item, weight in zip(collection, weights):
        r -= weight
        if r <= 0:
            return item
    assert False, "Could not sample from collection."


class NeuralNetwork:
    """Represents neural network implemented by a graph.

    Allows to create simple NeuralNetwork with capability to process an
    input and return output according to current weights, state and topology.

    Because artificial neural networks itself are represented as a graphs,
    there are cycles possible, so networks can be recursive after the
    training process.

    By default NeuralNetwork is simple two-layer, fully connected network
    consisting of input and output layers and one bias neuron.

    :Example:
    
        # Create fully connected two-layer net with 3 input nodes, five output
        # nodes and one bias node.
        nn = NeuralNetwork(3, 5)

        # Calculate and print an output for specified input vector.
        print nn.put([1, 3, 1])

        # Draw an underlying graph representing connections between nodes.
        nn.draw()
    """

    INPUT_NODE = 0.0
    OUTPUT_NODE = 1.0
    HIDDEN_NODE = 0.5
    BIAS_NODE = -1.0

    def __dir__(self):
        return ['put', 'draw', 'copy', 'is_similar', 'INPUT_NODE',
                'OUTPUT_NODE', 'HIDDEN_NODE', 'BIAS_NODE',
                '_update_node', '__str__', '__repr__', '_repr'
                                                       'activate', 'ins', '_g',
                'parse', 'save', 'fully_connected',
                'from_graph', 'load', 'reset_memory', 'complexity', 'outs']

    def __init__(self, ins, outs, activate=tanh, add_bias=True):
        """Create NeuralNetwork instance.

        Allows to create a two-layer NeuralNetwork with no connections
        with optional bias node.

        :param ins: number of network inputs
        :param outs: number of network outputs
        :param activate: activation function for neurons
        :param add_bias: if true, adds the bias node
        """

        self.activate = activate
        self.ins = ins
        self.outs = outs
        self._g = nx.DiGraph()
        self._g.add_nodes_from(range(1, ins + 1),
                               type=NeuralNetwork.INPUT_NODE,
                               updated=0, value=0.0)
        self._g.add_nodes_from(range(ins + 1, ins + outs + 1),
                               type=NeuralNetwork.OUTPUT_NODE,
                               updated=0,
                               value=0.0)
        if add_bias:
            self._g.add_node(ins + outs + 1,
                             type=NeuralNetwork.BIAS_NODE, updated=1,
                             value=1.0)
        self._repr = None

    def copy(self):
        """Copy NeuralNetwork."""

        return NeuralNetwork.from_graph(self.ins, self.outs,
                                        activate=self.activate,
                                        graph=self._g)

    def put(self, in_vec, auto_reset=True):
        """Calculate outputs for given input vector.

        Calculate output vector. May influence future calls if the network
        is recursive.
        :param in_vec: inputs
        :param auto_reset: if true, node states are brought to zero
        :return: network outputs
        """
        self._repr = None
        if len(in_vec) != self.ins:
            raise ValueError('Wrong number of inputs.')
        if auto_reset:
            self.reset_memory()
        in_nodes = [n for n in self._g if
                    self._g.node[n]['type'] == NeuralNetwork.INPUT_NODE]
        for n in self._g:
            if self._g.node[n]['type'] != NeuralNetwork.BIAS_NODE:
                self._g.node[n]['updated'] = 0
        for n, val in zip(in_nodes, in_vec):
            self._g.node[n]['value'] = val
            self._g.node[n]['updated'] = 1
        return [self._update_node(n) for n, node_attribute in
                self._g.nodes(data=True) if
                node_attribute['type'] == NeuralNetwork.OUTPUT_NODE]

    def is_similar(self, net, c1=1.0, c2=1.0, c3=0.4, d1=3.0):
        """States whether two nets are similar"""
        common = 0
        difference_occurred = False
        local_max_innovation = 0
        net_max_innovation = 0

        def _edges(n):
            return n._g.edges(data=True)

        def _feature_extractor(name):
            def _feats(*args):
                if len(args) == 1:
                    return args[0][2][name]
                return tuple(edge_info[2][name] for edge_info in args)

            return _feats

        innovs1 = sorted(
            map(_feature_extractor('innovation'),
                _edges(self)))
        innovs2 = sorted(
            map(_feature_extractor('innovation'),
                _edges(net)))
        innovs = map(None, innovs1, innovs2)
        for innov1, innov2 in innovs:
            if not difference_occurred and innov1 == innov2:
                common += 1
            else:
                difference_occurred = True
            if local_max_innovation < innov1:
                local_max_innovation = innov1
            if net_max_innovation < innov2:
                net_max_innovation = innov2
        lower_max = min(local_max_innovation, net_max_innovation)
        sel = lambda both_innovation_numbers: both_innovation_numbers[
            lower_max == local_max_innovation]
        excess = 0
        for innov in reversed(innovs):
            if sel(innov) and sel(innov) > lower_max:
                excess += 1
            elif sel(innov) <= lower_max:
                break
        all_genes = len(_edges(self)) + len(_edges(net))
        disjoint = all_genes - 2 * common - excess
        N = max(len(_edges(self)), len(_edges(net)), 1)
        weight_pairs = map(_feature_extractor('weight'),
                           _edges(self)[0:common], _edges(net)[0:common])
        differences = [(w1 - w2) ** 2 for w1, w2 in weight_pairs]
        if len(differences) > 0:
            W = sqrt(np.mean(differences))
        else:
            W = 0
        return (c1 * excess) / N + (c2 * disjoint) / N + c3 * W <= d1

    def _update_node(self, node):
        if self._g.node[node]['updated']:
            return self._g.node[node]['value']
        self._g.node[node]['updated'] = 1
        in_edges = self._g.in_edges(node, data=True)
        weights = [e_data['weight'] * e_data['active'] for src, dest, e_data in
                   in_edges]
        vals = [self._update_node(src) for src, dest, e_data in in_edges]
        self._g.node[node]['value'] = self.activate(
            sum(val * weight for val, weight in zip(vals, weights)))
        return self._g.node[node]['value']

    def reset_memory(self):
        """Resets network memory"""
        self._repr = None
        for n in self._g:
            if self._g.node[n]['type'] != NeuralNetwork.BIAS_NODE:
                self._g.node[n]['value'] = 0.0

    def draw(self):
        """Draws network"""
        # nx.draw_shell(self._g,node_color=[self._g.node[n]['type']
        # for n in self._g])

        # nx.write_dot(self._g,'tmp.dot')

        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(self._g, prog='dot')
            nx.draw(self._g, pos, with_labels=True, arrows=True,
                    node_color=[self._g.node[n]['type'] for n in
                                self._g])
        except KeyError:
            warn("graphviz plotting failed. Used draw_shell instead.")
            nx.draw_shell(self._g, with_labels=True, arrows=True,
                          node_color=[self._g.node[n]['type'] for n in
                                      self._g])

            # nx.draw_graphviz(self._g,with_labels=True,arrows=True)

    def complexity(self):
        """Returns a measure of complexity of a network"""
        def wt(edge):
            return edge[2]['weight']

        weights_complexity = sqrt(sum(
            wt(e) ** 2 for e in self._g.edges(data=True)))
        topology_complexity = len(self._g.edges())
        return topology_complexity + weights_complexity

    def __str__(self):
        s = ''
        for src, dst, data in self._g.edges(data=True):
            s += '(' + str(src) + ',' + str(dst) + ')'
            if data:
                s += '\n{\n'
                for key in data:
                    s += '\t' + str(key) + '=' + str(data[key]) + '\n'
                s += '}\n'
        return s

    def __repr__(self):
        if self._repr:
            return self._repr
        res = "ins %d\n" % self.ins
        res += "outs %d\n" % self.outs
        for line in nx.generate_gml(self._g):
            res += "%s\n" % line
        return res

    @staticmethod
    def parse(str_repr):
        """Parses network from string representation"""
        lines = str_repr.splitlines()
        ins = int(lines[0].split()[1])
        outs = int(lines[1].split()[1])
        graph = nx.parse_gml(lines[2:])
        return NeuralNetwork.from_graph(ins=ins, outs=outs, graph=graph)

    def save(self, path):
        """Saves network to file"""
        with open(path, 'w') as out:
            out.write(self.__repr__())

    @staticmethod
    def load(path):
        """Loads network from file"""
        with open(path) as f:
            return NeuralNetwork.parse(f.read())

    @staticmethod
    def fully_connected(ins, out, activate=tanh, add_bias=True):
        """Creates fully connected network"""
        net = NeuralNetwork(ins, out, activate=activate, add_bias=add_bias)
        innovation_number = 1
        for i in range(1, ins + 1):
            for j in range(ins + 1, ins + out + 1):
                net._g.add_edge(i, j,
                                active=1.0,
                                weight=rng.uniform(-1.0, 1.0),
                                sigma=rng.uniform(0.0, 1.0),
                                innovation=innovation_number)
                innovation_number += 1
        if add_bias:
            for j in range(ins + 1, ins + out + 1):
                net._g.add_edge(ins + out + 1, j, active=1.0,
                                weight=rng.uniform(-1.0, 1.0),
                                sigma=rng.uniform(0.0, 1.0),
                                innovation=innovation_number)
                innovation_number += 1
        return net

    @staticmethod
    def from_graph(ins, outs, graph, activate=tanh):
        """Creates NeuralNetwork from a graph"""
        net = NeuralNetwork(ins, 0, activate=activate)
        net.outs = outs
        if not isinstance(graph, nx.DiGraph):
            raise TypeError('Expected DiGraph.')
        net._g = graph.copy()
        return net


class Neat(param.Parameterized):
    """Class which hold the information abour NEAT algorithm progress."""

    _SEPARATOR = "=====================================================\n"

    _iteration = param.Integer(default=0, bounds=(0, float("inf")),
                               doc="Number of iterations passed.")
    weights_mutation_probability = param.Number(default=0.8, doc="Probability"
                                                                 " that weight"
                                                                 " will be "
                                                                 "randomly "
                                                                 "changed.",
                                                bounds=(0, 1))
    weights_perturbation_rate = param.Number(default=0.9, doc="Probability "
                                                              "that weight "
                                                              "will be "
                                                              "perturbed "
                                                              "normally from "
                                                              "its current "
                                                              "value, it will "
                                                              "be assigned "
                                                              "absolutely "
                                                              "random value "
                                                              "otherwise.",
                                             bounds=(0, 1))
    node_add_mutation_probability = param.Number(default=0.03, bounds=(0, 1),
                                                 doc="Probability that a new "
                                                     "node will be added.")
    connection_add_mutation_probability = param.Number(default=0.05,
                                                       bounds=(0, 1),
                                                       doc="Probability that a"
                                                           " new connection "
                                                           "will be added.")
    min_size_for_champion_selection = param.Integer(default=5, doc="Minimal"
                                                                   "size of "
                                                                   "species "
                                                                   "for which "
                                                                   "champions "
                                                                   "will be "
                                                                   "selected "
                                                                   "to "
                                                                   "persist.",
                                                    bounds=(0, float("inf")))
    elite_ratio = param.Number(default=0.4, bounds=(0, 1), doc="Rate of best "
                                                               "individuals in"
                                                               " the species, "
                                                               "that will be "
                                                               "considered as "
                                                               "elite.")
    allowed_staleness = param.Integer(default=15, doc="The number of "
                                                      "iterations after which "
                                                      "species are not allowed"
                                                      " to reproduce, if their"
                                                      " fitness did not "
                                                      "improve at all.",
                                      bounds=(0, float("inf")))
    species_mixing_ratio = param.Number(default=0.01, bounds=(0, 1),
                                        doc="Probability that inter-species "
                                            "crossover occurs.")
    c1 = param.Number(default=1.0, bounds=(0, float("inf")),
                      doc="Importance of excess genes. Used for similarity.")
    c2 = param.Number(default=1.0, bounds=(0, float("inf")),
                      doc="Importance of disjoint genes. Used for similarity.")
    c3 = param.Number(default=0.4, bounds=(0, float("inf")),
                      doc="Importance of weights difference in similarity.")
    d1 = param.Number(default=3.0, bounds=(0, float("inf")),
                      doc="Similarity threshold.")
    complexity_penalty = param.Number(default=0.0, bounds=(0, 1),
                                      doc="Penalty applied for model "
                                          "complexity as a percent of maximal "
                                          "error possible.")

    def __dir__(self):
        return ['_mutate_weights', '_mutate_add_node',
                '_mutate_add_connection',
                '_select_champions', '_select_elite',
                '_update_spec_fitness', '_crossover',
                '_recalculate_sizes', '_assign_spec',
                '_generate_offspring', 'evolve',
                'population_size', 'best', 'best_fitness',
                '_innovation_number', '_population',
                '_species', '_fitness', '_species_fitness',
                '_change_iter', '_iteration',
                'weights_mutation_probability', 'weights_perturbation_rate',
                'node_add_mutation_probability', 'complexity_penalty',
                'connection_add_mutation_probability',
                'min_size_for_champion_selection', 'elite_ratio',
                'allowed_staleness', 'species_mixing_ratio', 'c1', 'c2', 'c3',
                'd1', '__repr__', 'parse', '_repr', '_max_complexity',
                '_min_complexity', '_find_complexity_penalty']

    def __init__(self, inputs, outputs, population_size=100, net_factory=None):
        if not net_factory:
            def net_factory(ins, outs):
                """Factory creating empty networks"""
                # self._innovation_number = 1
                net = NeuralNetwork(ins, outs)
                # self._mutate_add_connection(net,probability=1.0)
                return net
        self.population_size = population_size
        self._innovation_number = (inputs + 1) * outputs + 1
        self._population = [net_factory(inputs, outputs) for i in
                            range(population_size)]
        compl = map(lambda n: n.complexity(), self._population)
        if len(compl) == 0:
            compl = [0]
        self._max_complexity, self._min_complexity = max(compl), min(compl)
        self._species = [0 for _ in range(population_size)]
        self._fitness = [None for _ in range(population_size)]
        self._species_fitness = {0: None}
        self._change_iter = {0: 0}
        self.best = None
        self.best_fitness = None
        self._repr = None

    def _mutate_weights(self, net, probability=0.8, perturbation_rate=0.9):
        self._repr = None
        if not isinstance(net, NeuralNetwork):
            raise TypeError('Expected NeuralNetwork.')
        # genetic algorithm variant - random new value
        if rng.uniform(0.0, 1.0) > perturbation_rate:
            for n in net._g:
                for e in net._g.edge[n]:
                    if rng.uniform(0.0, 1.0) < probability:
                        net._g.edge[n][e]['weight'] = rng.uniform(-1.0, 1.0)
                        net._g.edge[n][e]['sigma'] = rng.uniform(0.0, 1.0)
        # evolutionary strategy variant - gaussian perturbation
        else:
            if rng.uniform(0.0, 1.0) < probability:
                edges_no = len(net._g.edges())
                edges_no = edges_no if edges_no > 0 else 1
                t1 = 1 / sqrt(2 * edges_no)
                t2 = 1 / sqrt(2 * sqrt(edges_no))
                r1 = rng.gauss(0, 1)
                for src, dst in net._g.edges():
                    r2 = rng.gauss(0, 1)
                    net._g.edge[src][dst]['weight'] = \
                        rng.gauss(net._g.edge[src][dst]['weight'],
                                  net._g.edge[src][dst]['sigma'])
                    net._g.edge[src][dst]['sigma'] = \
                        net._g.edge[src][dst]['sigma'] * exp(r1 * t1) * exp(
                            r2 * t2)

    def _mutate_add_node(self, net, probability=0.03):
        if not isinstance(net, NeuralNetwork):
            raise TypeError('Expected NeuralNetwork.')

        if not net._g.edges():
            return

        new_edges = []
        new_node_num = len(net._g) + 1
        new_nodes = []

        if rng.uniform(0.0, 1.0) < probability:
            self._repr = None
            src, dst = rng.sample(net._g.edges(), 1)[0]
            net._g.edge[src][dst]['active'] = 0.0
            new_nodes.append(
                (new_node_num, NeuralNetwork.HIDDEN_NODE, False, 0.0))
            new_edges.append((src, new_node_num, 1.0,
                              rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0),
                              self._innovation_number))
            new_edges.append((new_node_num, dst, 1.0,
                              rng.uniform(-1.0, 1.0), rng.uniform(0.0, 1.0),
                              self._innovation_number + 1))
            new_node_num += 1
            self._innovation_number += 2

        for n, node_type, updated, value in new_nodes:
            net._g.add_node(n, type=node_type, updated=updated, value=value)
        for src, dst, active, weight, sigma, innovation in new_edges:
            net._g.add_edge(src, dst, active=active, weight=weight,
                            sigma=sigma, innovation=innovation)

    def _mutate_add_connection(self, net, probability=0.05):
        if not isinstance(net, NeuralNetwork):
            raise TypeError('Expected NeuralNetwork.')
        if rng.uniform(0.0, 1.0) > probability:
            return
        self._repr = None
        done = False
        unchecked_sources = [n for n in net._g]
        while not done and unchecked_sources:
            src = rng.sample(unchecked_sources, 1)[0]
            unchecked_sources.remove(src)
            successors = net._g.successors(src)
            if len(successors) < len(net._g) - net.ins - 1:
                dst = [n for n, data in net._g.nodes(data=True)
                       if n not in successors and n > net.ins
                       and data['type'] != NeuralNetwork.BIAS_NODE]
                dst = rng.sample(dst, 1)[0]
                weight = rng.uniform(-1.0, 1.0)
                sigma = rng.uniform(0.0, 1.0)
                active = 1.0
                innovation = self._innovation_number
                self._innovation_number += 1
                done = True
        if done:
            net._g.add_edge(src, dst, active=active, weight=weight,
                            sigma=sigma, innovation=innovation)

    def _select_champions(self, spec_min_size=5):
        assert isinstance(spec_min_size, (int, float, long)), \
            "spec_min_size must be numeric value."
        assert spec_min_size >= 0, \
            "spec_min_size must be non negative."
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        fits = {}
        sizes = {}
        champs = {}
        for i in range(self.population_size):
            spec = self._species[i]
            fit = self._fitness[i]
            if spec in fits:
                sizes[spec] += 1
                if fit > fits[spec]:
                    fits[spec] = fit
                    champs[spec] = i
            else:
                fits[spec] = fit
                sizes[spec] = 1
                champs[spec] = i
        return _unzip([(self._population[champs[key]],
                        self._species[champs[key]],
                        self._fitness[champs[key]])
                       for key in champs if sizes[key] > spec_min_size])

    def _select_elite(self, best_rate=0.4, allowed_staleness=15):
        assert isinstance(best_rate, (int, float, long)), \
            "best_rate must be numeric value."
        assert 0.0 <= best_rate <= 1.0, \
            "best_rate must be from range [0,1]."
        assert isinstance(allowed_staleness, (int, float, long)), \
            "allowed_staleness must be numeric value."
        assert allowed_staleness >= 0, \
            "allowed_staleness must be non negative."
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        specs_representants = {}
        decr_by_fitness = lambda individual: -individual[1]
        for i in range(self.population_size):
            spec = self._species[i]
            if self._change_iter[spec] - self._iteration > allowed_staleness:
                continue
            if spec in specs_representants:
                specs_representants[spec].append(
                    (self._population[i], self._fitness[i]))
            else:
                specs_representants[spec] = [
                    (self._population[i], self._fitness[i])]
        for key in specs_representants:
            s = sorted(specs_representants[key], key=decr_by_fitness)
            to_persist = int(ceil(best_rate * len(specs_representants[key])))
            assert to_persist > 0, "Estimated elite size zero."
            specs_representants[key] = s[0:to_persist]
        return specs_representants

    def _update_spec_fitness(self):
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        for i in range(self.population_size):
            spec = self._species[i]
            fit = self._fitness[i]
            if spec not in self._species_fitness or self._species_fitness[
                    spec] < fit:
                self._species_fitness[spec] = fit
                self._change_iter[spec] = self._iteration

    def _crossover(self, net1, net2, fit1, fit2):
        # we take fitter and add all non existing edges from worse
        if fit1 < fit2:
            return self._crossover(net2, net1, fit2, fit1)
        assert fit2 <= fit1, "Network fitness assumption failed."
        net = net1.copy()
        assert net is not net1, "Copied pointer instead of data."
        for src, dst, data in net2._g.edges(data=True):
            d = data
            # merging disjoint and excess genes only in the case of the same
            #  fitness
            if fit1 == fit2 and src not in net._g or dst not in \
                    net._g[src]:
                if src not in net._g:
                    node_data = \
                        [data for n, data in net2._g.nodes(data=True) if
                         n == src][0]
                    assert 'value' in node_data, "There was no node value in" \
                                                 " the node data."
                    assert 'updated' in node_data, "There was no update info" \
                                                   " in the node data."
                    net._g.add_node(src, node_data)
                if dst not in net._g:
                    node_data = \
                        [data for n, data in net2._g.nodes(data=True) if
                         n == dst][0]
                    assert 'value' in node_data, "There was no node value in" \
                                                 " the node data."
                    assert 'updated' in node_data, "There was no update info" \
                                                   " in the node data."
                    net._g.add_node(dst, node_data)
                assert 'weight' in d, "There was no weight in the edge data."
                assert 'active' in d, "There was no state of the edge (" \
                                      "activation) in the edge data."
                assert 'sigma' in d, "There was no sigma in the edge data."
                assert 'innovation' in d, "There was no historical " \
                                          "marking in the edge data."
                net._g.add_edge(src, dst, d)
            # mixing values from common nodes
            elif rng.uniform(0.0, 1.0) > 0.5:
                assert 'weight' in d, "There was no weight in the edge data."
                assert 'active' in d, "There was no state of the edge (" \
                                      "activation) in the edge data."
                assert 'sigma' in d, "There was no sigma in the edge data."
                assert 'innovation' in d, "There was no historical " \
                                          "marking in the edge data."
                net._g[src][dst]['weight'] = d['weight']
                net._g[src][dst]['active'] = d['active']
        self._mutate_weights(net,
                             probability=self.weights_mutation_probability,
                             perturbation_rate=self.weights_perturbation_rate)
        self._mutate_add_connection(net,
                                    probability=self.connection_add_mutation_probability)
        self._mutate_add_node(net,
                              probability=self.node_add_mutation_probability)
        return net

    def _recalculate_sizes(self, offspring_size, allowed_staleness=15):
        assert isinstance(allowed_staleness, (int, float, long)), \
            "allowed_staleness must be numeric value."
        assert allowed_staleness >= 0, \
            "allowed_staleness must be non negative."
        assert isinstance(offspring_size, (int, float, long)), \
            "offspring_size must be numeric value."
        assert 0 <= offspring_size <= self.population_size, \
            "offspring_size must be non negative and not greater than " \
            "population size."
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        min_fit, max_fit = min(self._fitness), max(self._fitness)
        occupied_range = max_fit - min_fit
        translation = -min_fit + 0.1 * (
            0.1 if not occupied_range else occupied_range)
        spec_sizes = {}
        spec_fit = {}
        for spec, fit in zip(self._species, self._fitness):
            if self._change_iter[spec] - self._iteration > allowed_staleness:
                continue
            if spec in spec_sizes:
                assert spec in spec_fit, "Species were added to spec_sizes " \
                                         "while not to spec_fit."
                spec_sizes[spec] += 1
                spec_fit[spec] += translation + fit
            else:
                spec_sizes[spec] = 1
                spec_fit[spec] = translation + fit
        adjusted_fitness_for_species = {spec: spec_fit[spec] / spec_sizes[spec]
                                        for spec in spec_fit}
        # reduced by common factor (counted individuals)
        mean_adjusted_fitness = sum(adjusted_fitness_for_species.values())
        scaling_factor = offspring_size
        # not rounding, but truncating (due to errors of too big offspring
        # size)
        estimates = {spec: int(scaling_factor * adjusted_fitness_for_species[
            spec] / mean_adjusted_fitness) for spec in spec_fit}
        corrections = 0
        spec_corrections = {}
        sum_of_estimates = sum(estimates.values())
        assert sum_of_estimates <= offspring_size, \
            "Sum of estimated exceeds offspring size."
        while sum_of_estimates + corrections < offspring_size:
            key = _weighted_sample(estimates, estimates.values())
            if key in spec_corrections:
                spec_corrections[key] += 1
            else:
                spec_corrections[key] = 1
            corrections += 1
        res = {key: estimates[key] + spec_corrections.get(key, 0) for key in
               estimates}
        assert sum(res.values()) == offspring_size, \
            "Estimated offspring size for species does not sum up to assumed" \
            " value."
        return res

    def _assign_spec(self, net):
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        species = None
        max_spec = -float("inf")
        for n, spec in zip(self._population, self._species):
            if net.is_similar(n, c1=self.c1, c2=self.c2, c3=self.c3,
                              d1=self.d1):
                species = spec
                break
            if max_spec < spec:
                max_spec = spec
        if species is None:
            species = max_spec + 1
        assert isinstance(species, (int, long)), \
            "Unknown type of species assigned."
        assert species >= 0, "Assigned species number should be non negative."
        return species

    def _generate_offspring(self, elites, offspring_size,
                            fitness_discriminator, mixing_ratio=0.01):
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        assert isinstance(offspring_size, (int, long, float)), \
            "offspring_size must be numeric value."
        assert 0 <= offspring_size <= self.population_size, \
            "offspring_size must be non-negative and not greater than " \
            "population size."
        assert isinstance(mixing_ratio, (int, long, float)), \
            "mixing_ratio must be numeric value."
        assert 0.0 <= mixing_ratio <= 1.0, \
            "mixing_ratio must be from range [0,1]."
        new_sizes = self._recalculate_sizes(offspring_size,
                                            allowed_staleness=
                                            self.allowed_staleness)
        offspring = []
        fits = []
        species = []
        for key in elites:
            assert key in new_sizes, "Elite must have sizes calculated."
            for i in range(new_sizes[key]):
                if rng.uniform(0.0, 1.0) <= mixing_ratio:
                    random_elite_key = rng.sample(elites, 1)[0]
                    parent2, fit2 = rng.sample(elites[random_elite_key], 1)[0]
                else:
                    parent2, fit2 = rng.sample(elites[key], 1)[0]
                assert isinstance(parent2,
                                  NeuralNetwork), "Both parents must be " \
                                                  "NeuralNetwork."
                assert isinstance(fit2, (
                    int, long,
                    float)), "Elites should be provided along with network " \
                             "fitness as numeric."
                parent1, fit1 = rng.sample(elites[key], 1)[0]
                assert isinstance(parent1,
                                  NeuralNetwork), "Both parents must be " \
                                                  "NeuralNetwork."
                assert isinstance(fit1, (
                    int, long,
                    float)), "Elites should be provided along with network " \
                             "fitness as numeric."
                net = self._crossover(parent1, parent2, fit1, fit2)
                assert isinstance(net,
                                  NeuralNetwork), "Crossover must return a " \
                                                  "NeuralNetwork."
                fit = fitness_discriminator(net)
                assert isinstance(fit, (int, long,
                                        float)), "Fitness function must " \
                                                 "return numeric value."
                spec = self._assign_spec(net)
                offspring.append(net)
                fits.append(fit)
                species.append(spec)
        return offspring, species, fits

    def _find_complexity_penalty(self, net):
        complexity = net.complexity()
        relative = complexity - self._min_complexity
        percentage = float(relative) / max(self._max_complexity, 1)
        scaled_by_factor = self.complexity_penalty * percentage
        if self._fitness[0] is not None:
            fit_range = max(self._fitness) - min(self._fitness)
        else:
            fit_range = 0
        scaled_by_fits = scaled_by_factor * fit_range
        # scaled_by_dims = scaled_by_factor*sqrt(net.outs)
        return scaled_by_fits

    def evolve(self, test_inputs=None, desired_outputs=None, net_fitness=None,
               goal=float("inf"), reset_memory=True):
        """Processes a single generation of networks"""
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        assert isinstance(goal, (int, long, float)), \
            "goal must be numeric value."
        self._repr = None
        if not net_fitness and test_inputs and desired_outputs:
            for item in test_inputs:
                assert isinstance(item, (
                    int, long, float)), "Inputs must be numeric values."
            for item in desired_outputs:
                assert isinstance(item, (
                    int, long,
                    float)), "Desired outputs must be numeric values."

            def _mse(outs):
                return sqrt(sum((e - o) ** 2 for o, e
                                in zip(outs, desired_outputs)))

            def net_fitness(net):
                """Network fitness function by means of negative MSE"""
                err = _mse(net.put(test_inputs, auto_reset=reset_memory))
                pen = self._find_complexity_penalty(net)
                return - err - pen
        assert net_fitness, "Inputs & desired outputs or fitness function " \
                            "must be provided."
        self._fitness = [net_fitness(indiv) for indiv in
                         self._population]
        for item in self._fitness:
            assert isinstance(item, (
                int, long,
                float)), "Fitness function must return numeric values."
        if max(self._fitness) >= goal:
            warn("Goal was already achieved.")
            self.best = None
            self.best_fitness = -float("inf")
            for net, fit in zip(self._population, self._fitness):
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best = net
            return self.best, self.best_fitness
        self._iteration += 1
        self._update_spec_fitness()
        persistent, pers_specs, pers_fits = self._select_champions(
            spec_min_size=self.min_size_for_champion_selection)
        for item in persistent:
            assert isinstance(item,
                              NeuralNetwork), "Champions should be " \
                                              "NeuralNetwork instances."
        for item in pers_specs:
            assert isinstance(item, (
                int, long)), "Champions' species should be of integer type."
            assert item >= 0, "Champions' species numbers should be " \
                              "non negative."
        for item in pers_fits:
            assert isinstance(item, (
                int, long,
                float)), "Champions' fitness should be numeric value."
        elites = self._select_elite(best_rate=self.elite_ratio,
                                    allowed_staleness=self.allowed_staleness)
        for key in elites:
            for parent, fit in elites[key]:
                assert isinstance(parent,
                                  NeuralNetwork), "Elites must provide " \
                                                  "NeuralNetwork instances."
                assert isinstance(fit, (int, long,
                                        float)), "Elites must provide " \
                                                 "numeric fitness values."
        offspring_size = self.population_size - len(persistent)
        offspring, offspring_specs, offspring_fitness = \
            self._generate_offspring(elites, offspring_size, net_fitness,
                                     mixing_ratio=self.species_mixing_ratio)
        for item in offspring:
            assert isinstance(item,
                              NeuralNetwork), "Offspring should consist of " \
                                              "NeuralNetwork instances."
        for item in offspring_specs:
            assert isinstance(item, (
                int, long)), "Offspring species should be of integer type."
        for item in offspring_fitness:
            assert isinstance(item, (int, long,
                                     float)), "Offspring must provide " \
                                              "numeric fitness values."
        self._population = list(persistent) + offspring
        self._fitness = list(pers_fits) + offspring_fitness
        self._species = list(pers_specs) + offspring_specs
        compl = map(lambda n: n.complexity(), self._population)
        self._max_complexity, self._min_complexity = max(compl), min(compl)
        self.best = None
        self.best_fitness = -float("inf")
        for net, fit in zip(self._population, self._fitness):
            if fit > self.best_fitness:
                self.best_fitness = fit
                self.best = net
        assert isinstance(self.best,
                          NeuralNetwork), "Best network should be instance " \
                                          "of NeuralNetwork."
        assert isinstance(self.best_fitness, (
            int, long, float)), "Best fitness must be numeric value."
        assert len(self._species) == self.population_size, \
            "Species vector is shorter than assumed population size."
        assert len(self._fitness) == self.population_size, \
            "Fitness vector is shorter than assumed population size."
        assert len(self._population) == self.population_size, \
            "Individuals vector is shorter than assumed population size."
        return self.best.copy(), self.best_fitness

    def __repr__(self):
        if self._repr:
            return self._repr
        str_repr = ""
        str_repr += "population_size %d\n" % self.population_size
        str_repr += "_innovation_number %d\n" % self._innovation_number
        str_repr += "_species %s\n" % self._species
        str_repr += "_fitness %s\n" % self._fitness
        str_repr += "_species_fitness %s\n" % self._species_fitness
        str_repr += "_change_iter %s\n" % self._change_iter
        str_repr += "_iteration %d\n" % self._iteration
        str_repr += "weights_mutation_probability %s\n" % \
                    self.weights_mutation_probability
        str_repr += "weights_perturbation_rate %s\n" % \
                    self.weights_perturbation_rate
        str_repr += "node_add_mutation_probability %s\n" % \
                    self.node_add_mutation_probability
        str_repr += "connection_add_mutation_probability %s\n" % \
                    self.connection_add_mutation_probability
        str_repr += "min_size_for_champion_selection %s\n" % \
                    self.min_size_for_champion_selection
        str_repr += "elite_ratio %s\n" % self.elite_ratio
        str_repr += "allowed_staleness %s\n" % self.allowed_staleness
        str_repr += "species_mixing_ratio %s\n" % self.species_mixing_ratio
        str_repr += "c1 %s\n" % self.c1
        str_repr += "c2 %s\n" % self.c2
        str_repr += "c3 %s\n" % self.c3
        str_repr += "d1 %s\n" % self.d1
        for net in self._population:
            str_repr += "%s%s\n" % (Neat._SEPARATOR, net.__repr__())
        return str_repr

    @staticmethod
    def parse(str_repr):
        """Parses whole population from string"""
        nt = Neat(inputs=0, outputs=0, population_size=0)
        parts = str_repr.split(Neat._SEPARATOR)
        local = parts[0]
        local = local.splitlines()
        for line in local:
            name = line.split()[0]
            value = line[len(name) + 1:]
            setattr(nt, name, ast.literal_eval(value))
        for net in parts[1:]:
            nt._population.append(NeuralNetwork.parse(net))
        max_fit = max(nt._fitness)
        nt.best, nt.best_fitness = [(net, fit) for net, fit in zip(
            nt._population, nt._fitness) if fit == max_fit][0]
        return nt

    def save(self, path):
        """Saves population to file"""
        with open(path, 'w') as out:
            out.write(self.__repr__())

    @staticmethod
    def load(path):
        """Loads population from file"""
        with open(path) as f:
            return Neat.parse(f.read())


if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Runs samples for the neat library.")
    parser.add_argument("--headless", dest='headless', const=True,
                        default=False,
                        action="store_const", help='Runs the test in the \
                        headless mode (with no drawing).')
    args = parser.parse_args()

    if args.headless:
        print "Running in headless mode. No figures will be displayed."

    print "NeuralNetwork test."
    rng.seed(0)
    nn = NeuralNetwork(3, 5)
    print nn.put([1, 3, 1])
    if not args.headless:
        nn.draw()

    print "Neat test: training population for XOR."
    rng.seed(0)
    neat = Neat(inputs=2, outputs=1, population_size=200)
    inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    outputs = [[-1.0], [1.0], [1.0], [-1.0]]


    def net_fitness(net):
        sum_of_squares = 0
        for idx in range(len(inputs)):
            sum_of_squares += sum((e - o) ** 2 for e, o in
                                  zip(outputs[idx], net.put(inputs[idx])))
        return -sqrt(sum_of_squares)


    for i in tqdm(range(100)):
        best, fitness = neat.evolve(net_fitness=net_fitness)

    print "Fitness: %s" % fitness

    if not args.headless:
        best.draw()
    assert net_fitness(best) == fitness, \
        "Calculated fitness is different than obtained from the evolve."
    for i in range(4):
        print "\tInput: %s, Output: %s" % (inputs[i], best.put(inputs[i]))
    print
