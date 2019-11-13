from assigment2 import GraphicalModel

g = GraphicalModel()

# g.load('small/asia.net')

# for node in g.factors:
#     print(g.factors[node])

# for node, value in g.outcomeSpace.items():
#     print(node, value)
g.insert('node 1', ('yes', 'no'))
g.insert('node 2', ('yes', 'no'))
g.connect('node 1', 'node 2')
g.factorize('node 1', [0.1, 0.9])
print(g.factors)
