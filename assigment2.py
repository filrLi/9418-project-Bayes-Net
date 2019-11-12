"""
Written by Zitong Li for COMP9418 assignment 2.
"""

import re
from graphviz import Digraph
from itertools import product
from collections import OrderedDict as odict


class GraphicalModel:
    def __init__(self):
        self.net = dict()
        self.factors = dict()
        self.outcomeSpace = dict()

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
                    parents = splits[1].split()
                data = [float(i) for i in re.findall(
                    r'[0-1][.][0-9]+', record[1])]
                self.factorize(node, parents, data)

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

    def factorize(self, node, parents, data):
        """
        Specify probabilities for a node.
        """
        dom = [node] + parents
        dom.reverse()
        dom = tuple(dom)
        for parent in parents:
            self.connect(parent, node)

        self.factors[node] = {'dom': dom, 'table': odict()}
        outcome_product = product(*[self.outcomeSpace[node] for node in dom])
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
