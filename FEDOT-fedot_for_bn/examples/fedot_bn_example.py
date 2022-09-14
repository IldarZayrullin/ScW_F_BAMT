from lib2to3.pytree import type_repr
from re import I
import sys
from typing import Optional, Union, List
parentdir = 'C:\\Users\\user\\Downloads\\FEDOT-fedot_for_bn'
sys.path.insert(0, parentdir)

from fedot.core.dag.graph import Graph
from joblib import PrintTime

from copy import deepcopy
import itertools
from fedot.core.dag.graph_node import GraphNode
 
import pandas as pd
import random
from functools import partial
from sklearn import preprocessing
import seaborn as sns
 
import bamt.Preprocessors as pp
from bamt.Builders import StructureBuilder
import bamt.Networks as Nets
 
from fedot.core.composer.gp_composer.gp_composer import PipelineComposerRequirements
from fedot.core.dag.validation_rules import has_no_cycle, has_no_self_cycled_nodes
from fedot.core.log import default_log, Log
from fedot.core.optimisers.adapters import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimiser import EvoGraphOptimiser, GPGraphOptimiserParameters, \
    GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.crossover import CrossoverTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum #этой нет в новом
from fedot.core.optimisers.gp_comp.operators.selection import SelectionTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.graph import OptGraph, OptNode
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score, BicScore, BDeuScore
from sklearn.metrics import mean_squared_error, accuracy_score #also
import matplotlib.pyplot as plt #also

# кастомный граф
# TODO find out Graph or OptGraph need to use
class CustomGraphModel(Graph):

    def __init__(self, nodes: Optional[Union[OptNode, List[OptNode]]] = None):
        super().__init__(nodes)
        # TODO find out is this field necessary or not - fix it
        self.unique_pipeline_id = 1


# кастомные узлы
class CustomGraphNode(OptNode):
    def __str__(self):
        return self.content["name"]
 
# кастомная метрика
def custom_metric(graph: CustomGraphModel, data: pd.DataFrame, method = 'K2'):
    score = 0
    nodes = data.columns.to_list()
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        # if 'Node' in l1:
        #     l1 = l1.split('_')[1]
        # if 'Node' in l2:
        #     l2 = l2.split('_')[1]
        struct.append([l1, l2])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)
    if method == 'K2':
        score = K2Score(data).score(bn_model)
    elif method == 'Bic':
        score = BicScore(data).score(bn_model)
    elif method == 'BDeu':
        score = BDeuScore(data).score(bn_model)
    else:
        print('No such method. Select K2, Bic or BDeu')
    #score = score + nodes_have
    return [score]

def opt_graph_to_bamt(graph: CustomGraphModel):
    graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = []
    for pair in graph_nx.edges():
        l1 = str(labels[pair[0]])
        l2 = str(labels[pair[1]])
        if 'Node' in l1:
            l1 = l1.split('_')[1]
        if 'Node' in l2:
            l2 = l2.split('_')[1]
        struct.append((l1, l2))
    return struct 
 

# проверка "нет дубликатов узлов"
def _has_no_duplicates(graph):
    _, labels = graph_structure_as_nx_graph(graph)
    if len(labels.values()) != len(set(labels.values())):
        raise ValueError('Custom graph has duplicates')
    return True
 
 
def _has_disc_parents(graph):
    graph, labels = graph_structure_as_nx_graph(graph)
    for pair in graph.edges():
        if (node_type[str(labels[pair[1]])] == 'disc') & (node_type[str(labels[pair[0]])] == 'cont'):
            raise ValueError(f'Discrete node has cont parent')
    return True

def _no_empty_graph(graph):
    graph, _ = graph_structure_as_nx_graph(graph)
    if len(graph.edges())==0:
        raise ValueError(f'Graph empty')
    return True
 

def custom_crossover_parents(graph_first, graph_second, max_depth):
    num_cros = 1
    try:
        for _ in range(num_cros):
            new_graph_first=deepcopy(graph_first)

            dir_of_nodes={new_graph_first.nodes[i].content['name']:i for i in range(len(new_graph_first.nodes))}
            edges = graph_second.operator.get_all_edges()
            flatten_edges = list(itertools.chain(*edges))
            nodes_with_parent_or_child=list(set(flatten_edges))
            # защита от пустых графов
            if nodes_with_parent_or_child!=[]:

                selected_node=random.choice(nodes_with_parent_or_child)
                parents=selected_node.nodes_from

                node_from_first_graph=new_graph_first.nodes[dir_of_nodes[selected_node.content['name']]]
            
                node_from_first_graph.nodes_from=[]
                if parents!=[] and parents!=None:
                    parents_in_first_graph=[new_graph_first.nodes[dir_of_nodes[i.content['name']]] for i in parents]
                    for i in range(len(parents_in_first_graph)):
                        node_from_first_graph.nodes_from.append(parents_in_first_graph[i])
    except Exception as ex:
        graph_first.show()
        print(ex)
 
    return new_graph_first, graph_second


    
    
#    [new_node.nodes_from.append(i) for i in parents]
    


    # pairs_of_nodes = equivalent_subtree(graph_first, graph_second)
    # if pairs_of_nodes:
    #     node_from_graph_first, node_from_graph_second = choice(pairs_of_nodes)

    #     layer_in_graph_first = \
    #         graph_first.root_node.distance_to_primary_level - node_from_graph_first.distance_to_primary_level
    #     layer_in_graph_second = \
    #         graph_second.root_node.distance_to_primary_level - node_from_graph_second.distance_to_primary_level

    #     replace_subtrees(graph_first, graph_second, node_from_graph_first, node_from_graph_second,
    #                      layer_in_graph_first, layer_in_graph_second, max_depth)
    return new_graph_first, graph_second


# кастомная мутация. ??? Здеь 10 раз пытается провести ориентированное ребро так, чтобы не появился цикл
def custom_mutation_add(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            nodes_not_cycling = (random_node.descriptive_id not in
                                 [n.descriptive_id for n in other_random_node.ordered_subnodes_hierarchy()] and
                                 other_random_node.descriptive_id not in
                                 [n.descriptive_id for n in random_node.ordered_subnodes_hierarchy()])
            # if other_random_node.nodes_from is not None and len(other_random_node.nodes_from) == 0:
            #     other_random_node.nodes_from = None
            if nodes_not_cycling:
                graph.operator.connect_nodes(random_node, other_random_node)
#                graph.show()

    except Exception as ex:
        graph.log.warn(f'Incorrect connection: {ex}')
    return graph
 

def custom_mutation_delete(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.disconnect_nodes(other_random_node, random_node, False)
    except Exception as ex:
        #graph.log.warn(f'Incorrect connection: {ex}')
        print(ex)
#        graph.show()
        # print(random_node.nodes_from is None)
        # print(other_random_node not in random_node.nodes_from)
        # print(other_random_node not in graph.nodes)
        # print(random_node not in graph.nodes)
        # print(not (random_node.nodes_from is None or other_random_node not in random_node.nodes_from 
        #         or other_random_node not in graph.nodes or random_node not in graph.nodes))



    return graph

def custom_mutation_reverse(graph: OptGraph, **kwargs):
    num_mut = 10
    try:
        for _ in range(num_mut):
            rid = random.choice(range(len(graph.nodes)))
            random_node = graph.nodes[rid]
            other_random_node = graph.nodes[random.choice(range(len(graph.nodes)))]
            if random_node.nodes_from is not None and other_random_node in random_node.nodes_from:
                graph.operator.reverse_edge(other_random_node, random_node)              
    except Exception as ex:
        #graph.log.warn(f'Incorrect connection: {ex}')
        print(ex)
    return graph

# главная функция, которая последовательно запускается
def run_example():
 
    # Импорт данных и дискретизация
    data=pd.read_csv(r'data/hack_processed_with_rf.csv')
    global vertices
    vertices = ['Tectonic regime', 'Period', 'Lithology',
                    'Structural setting', 'Gross', 'Netpay',
                    'Porosity', 'Permeability', 'Depth']
    dir_of_vertices={vertices[i]:i for i in range(len(vertices))}
    # vertices1 = ['Depth','Period', 'Lithology',
    #                 'Structural setting','Tectonic regime', 'Gross', 'Netpay',
    #                 'Porosity', 'Permeability']
    # dir_of_vertices={vertices1[i]:i for i in range(len(vertices1))}
    data = data[vertices]
    data.dropna(inplace=True)
    data.reset_index(inplace=True, drop=True)
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    global node_type
    #node_type=dict(('Node_'+key, value) for (key, value) in p.info['types'].items())
    node_type = p.info['types'] 
    bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    bn.add_nodes(p.info)
    bn.add_edges(discretized_data, scoring_function=('K2', K2Score))

    
    
 



 
    # правила: нет петель, нет циклов, нет дибликатов узлов
    rules = [has_no_self_cycled_nodes, has_no_cycle, _has_no_duplicates, _has_disc_parents]
    # изициализация графа без связей, только с узлами
    initial = [CustomGraphModel(nodes=[CustomGraphNode(nodes_from=[],
                                                      content={'name': v}) for v in vertices])]

    # Fedot example
    # node_scaling = PrimaryNode('scaling')
    # node_normalization = PrimaryNode('normalization')
    # node_linear = SecondaryNode('linear', nodes_from=[node_scaling, node_normalization])
    
    # Set nodes connections
    

    
    for node in initial[0].nodes: 
        parents = []
        for n in bn.nodes:
            if str(node) == str(n):
                parents = n.cont_parents + n.disc_parents
                break
        for n2 in initial[0].nodes:
            if str(n2) in parents:
                node.nodes_from.append(n2)

    # initial[0].show()

    #параметры ГА
    requirements = PipelineComposerRequirements(
        primary=vertices,
        secondary=vertices, max_arity=100,
        max_depth=100, pop_size=5, num_of_generations=5,
        crossover_prob=0.8, mutation_prob=0.9)
 
    # Ещё параметры для ГА
    # genetic_scheme_type -> [steady_state, generational, parameter_free]
    # crossover_types -> [subtree, one_point, none] CrossoverTypesEnum
    # mutation_types -> [simple, reduce, growth, local_growth] MutationTypesEnum
    # selection_types -> [tournament,nsga2, spea2] SelectionTypesEnum
    optimiser_parameters = GPGraphOptimiserParameters(
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
    # добавила селекцию турниром
        selection_types=[SelectionTypesEnum.tournament],
        mutation_types=[custom_mutation_add, custom_mutation_delete, custom_mutation_reverse],
        crossover_types=[custom_crossover_parents],
        regularization_type=RegularizationTypesEnum.none,
        stopping_after_n_generation=5
    )
 
    # Параметры для генерации графов. Закидываем сюда граф, узлы и правила
    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(base_graph_class=CustomGraphModel, base_node_class=CustomGraphNode),
        rules_for_constraint=rules)
 
    # Эволюционный оптимизатор графов
    optimiser = EvoGraphOptimiser(
        graph_generation_params=graph_generation_params,
        metrics=[],
        parameters=optimiser_parameters,
        requirements=requirements, initial_graph=initial,
        log=default_log(logger_name='Bayesian', verbose_level=1))
 
    # в partial указываем целевую функцию
    optimized_graph = optimiser.optimise(partial(custom_metric, data=discretized_data, method = met))
    # Сейчас optimized_graph хранит: {'depth': 9, 'length': 9, 'nodes': [Period, Netpay, ...]}
    optimized_graph.nodes=deepcopy(sorted(optimized_graph.nodes, key=lambda x: dir_of_vertices[x.content['name']]))
    # Отличается от optimized_graph только добавлением 'Node_' к названиям узлов
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graph)

    #optimized_network.show() C:/Users/user/Documents/FEDOT
    # print(round(custom_metric(optimized_network, method=met, data=discretized_data)[0],2))

    optimized_graph.show(path=('RESULTS\\V' + str(i) + '.png')) 
    OF=round(custom_metric(optimized_network, method=met, data=discretized_data)[0],2)
 
    #return opt_graph_to_bamt(optimized_network)
    
    final_bn = Nets.HybridBN(has_logit=False, use_mixture=False)
    final_bn.add_nodes(p.info)
    structure = opt_graph_to_bamt(optimized_network)
    print(structure)
    final_bn.set_structure(edges=structure)
    final_bn.get_info()
    final_bn.fit_parameters(data)

    prediction = dict()

    for c in vertices:
        test = data.drop(columns=[c])
        pred = final_bn.predict(test, 5)
        prediction.update(pred)
    
    result = dict()
    for key, value in prediction.items():
        if node_type[key]=="disc":
            res=round(accuracy_score(data[key], value),2)
        elif node_type[key]=="cont":
            res=round(mean_squared_error(data[key], value, squared=False),2)
        result[key]=res

    #r_for_pdf=pd.DataFrame.from_dict(result, orient='index', columns=['result'])
    
    pdf.add_page()
    pdf.set_font("Arial", size = 14)
    
    pdf.cell(150, 5, txt = 'Objective function = ' + str(met)+ ' = ' + str(OF),
            ln = 1, align = 'C')
    pdf.cell(150, 5, txt = "pop_size = " + str(requirements.pop_size),
            ln = 1)
    pdf.cell(150, 5, txt = "mutation_prob = " + str(requirements.mutation_prob),
            ln = 1)
    pdf.cell(150, 5, txt = "genetic_scheme_type = " + str(optimiser_parameters.genetic_scheme_type),
            ln = 1)
    pdf.cell(150, 5, txt = "selection_types = " + str(optimiser_parameters.selection_types),
            ln = 1)
    pdf.cell(150, 5, txt = "mutation_types = " + str(optimiser_parameters.mutation_types),
            ln = 1)
    pdf.cell(150, 5, txt = "crossover_types = " + str(optimiser_parameters.crossover_types),
            ln = 1)
    pdf.cell(150, 5, txt = "stopping_after_n_generation = " + str(optimiser_parameters.stopping_after_n_generation),
            ln = 1)
    pdf.cell(150, 5, txt = "actual_generation_num = " + str(optimiser.generation_num),
            ln = 1)            
    pdf.image('RESULTS\\V' + str(i) + '.png',w=165, h=165)
    pdf.multi_cell(180, 5, txt = str(structure))
    # pdf.cell(40, 10, txt =str(r_for_pdf), ln = 1)

    
    for heading, row in result.items():
        pdf.cell(40, 6, heading, 1)
        pdf.cell(40, 6, str(row), 1)
        pdf.ln()
    pdf.output("Check"+str(i)+".pdf")
    # print(pred)
    # print(prediction)
    # sample = final_bn.sample(400)
    # sns.histplot(binwidth=0.5, x="Tectonic regime", data=sample, stat="count")
    # plt.show()





if __name__ == '__main__':
    # data = pd.read_csv(r'examples/data/hack_processed_with_rf.csv')
    # nodes = ['Tectonic regime', 'Period', 'Lithology',
    #                 'Structural setting', 'Gross', 'Netpay',
    #                 'Porosity', 'Permeability', 'Depth']
    # data = data[nodes]

    from fpdf import FPDF


    for i, met in zip(range(1,3), ['BDeu','K2']):
        pdf = FPDF()
        structure = run_example()


    # for i, met in zip(range(1,4), ['K2', 'Bic', 'BDeu']):
    #     structure = run_example()

    
   



    # bn = Nets.HybridBN(has_logit=True, use_mixture=True)
    # encoder = preprocessing.LabelEncoder()
    # discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    # p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    # discretized_data, est = p.apply(data)
    # info = p.info
    # bn.add_nodes(info)
    # tmp = StructureBuilder(info)
    # tmp.skeleton = {
    #     'V': bn.nodes,
    #     'E': structure
    # }
    # tmp.get_family()
    # bn.nodes = tmp.skeleton['V']
    # bn.edges = tmp.skeleton['E']
    # print(bn.nodes)
    # print(bn.edges)
    # print(bn.get_info())
