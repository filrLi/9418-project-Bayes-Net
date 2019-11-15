"""
Written by Zitong Li for COMP9418 assignment 2.
"""

import re
import numpy as np
import pandas as pd
import copy
from graphviz import Digraph
from itertools import product
from collections import OrderedDict as odict


class GraphicalModel:
    def __init__(self):
        self.net = dict()
        self.factors = dict()
        self.outcomeSpace = dict()
        self.node_value = dict()

    def load(self, FileName):
        """
        Load and initiate model from file
        """
        with open(FileName, 'r') as f:
            content = f.read()
            node_pattern = re.compile(
                r'node (.+) \n\{\n  states = \( \"(.+)\" \);\n\}')
            potential_pattern = re.compile(
                r'potential \( (.+) \) \n\{\n  data = \((.+)\)[ ]?;\n\}')

            nodes_records = node_pattern.findall(content)
            data_records = potential_pattern.findall(content)

            for record in nodes_records:
                outcome = tuple(re.split(r'\" \"', record[1]))
                self.insert(record[0], outcome)
            for record in data_records:
                splits = record[0].split(' | ')
                node = splits[0]
                parents = []
                if len(splits) > 1:
                    parents = list(reversed(splits[1].split()))
                data = [float(i) for i in re.findall(
                    r'[0-1][.][0-9]+', record[1])]
                self.factorize(node, data, parents)

    def connect(self, father, child):
        """
        Connect Two nodes.
        """
        if father in self.net and child in self.net and child not in self.net[father]:
            self.net[father].append(child)

    def disconnect(self, father, child):
        """
        Disconnect Two nodes.
        """
        if father in self.net and child in self.net:
            self.net[father].remove(child)

    def factorize(self, node, data, parents=[]):
        """
        Specify probabilities for a node.
        data is a 1-d array or a simple list.
        """
        dom = parents + [node]
        dom = tuple(dom)
        for parent in parents:
            self.connect(parent, node)

        self.factors[node] = {'dom': dom, 'table': odict()}
        outcome_product = product(*[self.outcomeSpace[node] for node in dom])
        assert np.prod([len(self.outcomeSpace[node])
                        for node in dom]) == len(data), 'CPT length illegal'
        for i, combination in enumerate(outcome_product):
            self.factors[node]['table'][combination] = data[i]

    def insert(self, Name, Outcome):
        if Name not in self.net:
            self.net[Name] = []
            self.outcomeSpace[Name] = Outcome
        else:
            print(f'Already have node {Name}')

    def remove(self, node):
        if node in self.net:
            self.net.pop(node)
            self.outcomeSpace.pop(node)

    def save(self, fileName):
        f = open(fileName, 'w')
        f.write('net\n{\n}\n')

        # first node domain part
        for node, values in self.outcomeSpace.items():
            outcome = " ".join(
                ['"' + value + '"' for value in values])
            text = 'node %s \n{\n  states = (%s);\n}\n' % (node, outcome)
            f.write(text)

        # add data
        for node, relation in self.factors.items():
            potential = relation['dom'][0]
            data = " ".join(relation['table'].values())
            if len(relation['dom']) > 1:
                potential += ' | ' + " ".join(relation['dom'][1:])
                data = " "

        f.close()

    def showGraph(self):
        """
        Visualize the net graph.
        """
        dot = Digraph()
        dot.attr(overlap="False", splines="True")
        for v in self.net:
            dot.node(str(v))
        for v in self.net:
            for w in self.net[v]:
                dot.edge(str(v), str(w))
        return dot

    def prune(self, query, evidence=[]):
        qe = set(query + evidence)
        assert all([_ in self.net for _ in qe])
        newG = copy.deepcopy(self)
        all_deleted = 0

        while not all_deleted:
            all_deleted = 1
            W = set()
            for node, children in newG.net.items():
                if node not in qe and not children:
                    W.add(node)
                    all_deleted = 0
            for leaf in W:
                newG.remove(leaf)
            # clear the child who have been deleted
            for node, children in newG.net.items():
                newG.net[node] = [_ for _ in children if _ not in W]

        netcopy = copy.deepcopy(newG.net)
        for node in evidence:
            netcopy[node] = []

        reachable_from_q = self.spread(self.make_undirected(netcopy), query)
        nodes = list(newG.net.keys())
        for node in nodes:
            if node not in qe | reachable_from_q:
                newG.remove(node)
        return newG

    def spread(self, graph, source):
        visited = set()
        for node in source:
            self.spread_help(graph, node, visited)
        return visited

    def spread_help(self, graph, node, visited):
        visited.add(node)
        for child in graph[node]:
            if child not in visited:
                self.spread_help(graph, child, visited)

    def make_undirected(self, graph):
        undirectG = graph.copy()
        GT = self.transposeGraph(graph)
        for node in graph:
            undirectG[node] += GT[node]
        return undirectG

    def transposeGraph(self, G):
        GT = dict((v, []) for v in G)
        for v in G:
            for w in G[v]:
                if w in GT:
                    GT[w].append(v)
                else:
                    GT[w] = [v]
        return GT

    def gibbs_sampling(self, q_vars, sample_num, chain_num=2, **q_evis):
        prunned_graph = self.prune(q_vars, list(q_evis.keys()))
        chains = prunned_graph.burn_in(chain_num, q_evis)
        samples = []
        # fisrt sample
        sample = dict()
        for var in q_vars:
            sample[var] = chains[0].node_value[var]
        samples.append(sample)
        curr = 1
        while curr < sample_num:
            sample = dict()
            for var in q_vars:
                chain = chains[np.random.choice(chain_num)]
                pre_value = samples[curr - 1][var]
                value = chain.sample_once(var)
                A = chain.get_acceptance(var, pre_value, value)
                sample[var] = np.random.choice(
                    [value, pre_value], 1, p=[A, 1-A])[0]
            samples.append(sample)
            curr += 1
        return samples

    def get_acceptance(self, node, pre, curr):
        dom = self.factors[node]['dom']
        parents = dom[: -1]
        parents_value = [self.node_value[parent] for parent in parents]
        ppre = self.factors[node]['table'][tuple(parents_value + [pre])]
        pcurr = self.factors[node]['table'][tuple(parents_value + [curr])]
        return min(1, pcurr/ppre)

    def burn_in(self, chain_num, evidences, window_size=100):
        chains = []
        chains_non_evis = []
        for seed in range(chain_num):
            np.random.seed(seed)
            chain = copy.deepcopy(self)
            # 1. fix evidence
            chain.node_value = evidences.copy()
            # 2: Initialize other variables
            non_evis = dict()
            for node, domain in self.outcomeSpace.items():
                if node not in evidences:
                    value = np.random.choice(domain, 1)[0]
                    chain.node_value[node] = value
                    non_evis[node] = [value]
            chains.append(chain)
            chains_non_evis.append(non_evis)

        sample_count = 1
        while True:
            if sample_count >= window_size:
                if self.mixed(chains_non_evis, self.outcomeSpace):
                    break
                # clear the chains_non_evis
                chains_non_evis = [{
                    node: []
                    for node in chains_non_evis[i].keys()
                } for i in range(chain_num)]
                sample_count = 0

            # 3: Choose a variable ordering O
            O = np.random.permutation(list(chains_non_evis[0].keys()))
            # 4: Repeat sample non_evis in the order O
            for var in O:
                for i, chain in enumerate(chains):
                    value = chain.sample_once(var)
                    chain.node_value[var] = value
                    chains_non_evis[i][var].append(value)
                sample_count += 1
        return chains

    def sample_once(self, node):
        dom = self.factors[node]['dom']
        parents = dom[: -1]
        parents_value = [self.node_value[parent] for parent in parents]
        combinations = [tuple(parents_value + [node_value])
                        for node_value in self.outcomeSpace[node]]
        prob_list = [self.factors[node]['table'][combination]
                     for combination in combinations]
        return np.random.choice(self.outcomeSpace[node], 1, p=prob_list)[0]

    @staticmethod
    def convert(list_of_dict, outcomeSpace):
        mapping = dict()
        for node, values in outcomeSpace.items():
            mapping[node] = dict()
            for value in values:
                mapping[node][value] = (values.index(value)+1) / len(values)
        for i, record in enumerate(list_of_dict):
            list_of_dict[i] = {key: [mapping[key][value]
                                     for value in item] for key, item in record.items()}
        return list_of_dict

    def mixed(self, chain_vars, outcomeSpace):
        """
        to judge whether chain_vars are mixed up
        chain_vars = [
            {a:[...], b:[...] ...},
            {a:[...], b:[...] ...}]
        """
        # covert text value into num like value
        chain_vars = self.convert(chain_vars, outcomeSpace)

        parameters = list(chain_vars[0].keys())
        P_hat = []
        df_list = [pd.DataFrame(var_dic) for var_dic in chain_vars]
        concat_df = pd.concat(df_list, ignore_index=True)
        for parm in parameters:
            W = np.mean([df[parm].var() for df in df_list])
            B = concat_df[parm].var()
            P_hat.append((B / W) ** 0.5)
        return all([_ < 1.1 for _ in P_hat])
