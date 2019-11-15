import pandas as pd
from assigment2 import GraphicalModel

g = GraphicalModel()

g.load('small/asia.net')

# listD = [{'a': ['1', '2', '3'], 'b':['5', '4', '5']},
#          {'a': ['4', '2', '3'], 'b':['3', '6', '5']}]
# outcomespace = {'a': ['1', '2', '3', '4'], 'b': ['6', '5', '3', '4']}
# # p = g.convert(listD, outcomespace)
# # p0 = pd.DataFrame(p[0])
# # sp = sum(p0.var(axis=0))
# # print(p0.var(axis=0))
# # print(sp)
# a = g.mixed(listD, outcomespace)
# print(a)

samples = g.gibbs_sampling(['lung', 'bronc'], 50, tub='yes')
for sample in samples:
    print(sample)
