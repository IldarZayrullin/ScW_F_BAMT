import time

start = time.time()

from bamt.Preprocessors import Preprocessor
import pandas as pd
from sklearn import preprocessing as pp
import bamt.Networks as Networks
from bamt.utils import GraphUtils as gru

p1 = time.time()
print(f"Time elapsed for importing: {p1 - start}")

h = pd.read_csv("../Data/hack_processed_with_rf.csv")

cols = ['Tectonic regime', 'Period', 'Lithology', 'Structural setting', 'Gross','Netpay','Porosity','Permeability', 'Depth']
h = h[cols]

p2 = time.time()
print(f"Time elapsed for uploading data: {p2 - p1}")

encoder = pp.LabelEncoder()
discretizer = pp.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

p = Preprocessor([('encoder', encoder)])
discretized_data, est = p.apply(h)
info = p.info

# --------------------- VALIDATION TEST-------------------------------
nodes_type_mixed = gru.nodes_types(h)
columns = [col for col in h.columns.to_list() if nodes_type_mixed[col] in ['disc','disc_num']] # GET ONLY DISCRETE
discrete_data = h[columns]

discretized_data, est = p.apply(discrete_data) # warning
info = p.info

bn = Networks.HybridBN()
bn.add_nodes(descriptor=info) # error
# ------------------------------
p = Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
discretized_data, est = p.apply(h)
info = p.info
# ---------------------------------------
print("has_logit=False, use_mixture=False")
bn = Networks.HybridBN()
bn.add_nodes(descriptor=info)

for node in bn.nodes:
    print(f"{node.name}: {node.type}") # only gaussian and discrete nodes
print("#"*150)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))
# -----------------
print("has_logit=True, use_mixture=False")
bn = Networks.HybridBN(has_logit=True)
bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))
# --------------------------
print("has_logit=True, use_mixture=True")
bn = Networks.HybridBN(has_logit=True, use_mixture=True)
bn.add_nodes(descriptor=info)

bn.add_edges(data=discretized_data, optimizer='HC', scoring_function=('MI',))
t1 = time.time()
bn.fit_parameters(data=h)
t2 = time.time()
print(f'PL elaspsed: {t2-t1}')
bn.save_params("hack_p.json")
bn.save_structure("hack_s.json")
# for node, d in bn.distributions.items():
#     print(node)
#     for param, value in d.items():
#         print(f"{param}:{value}")
