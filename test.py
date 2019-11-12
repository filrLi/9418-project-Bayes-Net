from assigment2 import GraphicalModel

g = GraphicalModel()

g.load('small/asia.net')

# for node in g.factors:
#     print(g.factors[node])

# for node, value in g.outcomeSpace.items():
#     print(node, value)
g.save('text.net')
