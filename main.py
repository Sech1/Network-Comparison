import collections
import operator
import threading
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from community import community_louvain as cl


def main():
    FLAG = 6

    non_random_graph = open_file_and_parse()
    non_random_graph.name = 'Real Graph'
    barabasi_graph = nx.barabasi_albert_graph(4941, 2)
    barabasi_graph.name = 'Barabasi Graph'
    random_graph = nx.erdos_renyi_graph(4941, 0.0019, directed=False)
    random_graph.name = 'Random Graph'

    if FLAG == 0:
        thread1 = threading.Thread(target=plot_degree_distribution, args=(non_random_graph,))
        thread2 = threading.Thread(target=plot_degree_distribution, args=(barabasi_graph,))
        thread3 = threading.Thread(target=plot_degree_distribution, args=(random_graph,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    elif FLAG == 1:
        thread1 = threading.Thread(target=cluster_coefficient, args=(non_random_graph,))
        thread2 = threading.Thread(target=cluster_coefficient, args=(barabasi_graph,))
        thread3 = threading.Thread(target=cluster_coefficient, args=(random_graph,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    elif FLAG == 2:
        thread1 = threading.Thread(target=cluster, args=(non_random_graph,))
        thread2 = threading.Thread(target=cluster, args=(barabasi_graph,))
        thread3 = threading.Thread(target=cluster, args=(random_graph,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    elif FLAG == 3:
        thread1 = threading.Thread(target=find_cluster_table, args=(non_random_graph,))
        thread2 = threading.Thread(target=find_cluster_table, args=(barabasi_graph,))
        thread3 = threading.Thread(target=find_cluster_table, args=(random_graph,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    elif FLAG == 4:
        thread1 = threading.Thread(target=betweenness_centrality, args=(non_random_graph,))
        thread2 = threading.Thread(target=betweenness_centrality, args=(barabasi_graph,))
        thread3 = threading.Thread(target=betweenness_centrality, args=(random_graph,))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    elif FLAG == 5:
        real_calculations(non_random_graph)
    elif FLAG == 6:
        find_node(non_random_graph)


def plot_degree_distribution(graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    print(str(graph.name) + " Nodes: " + str(graph.number_of_nodes()) + '\n')
    print(str(graph.name) + " Edges: " + str(graph.number_of_edges()) + '\n')

    if graph.name == 'Random Graph':
        print(str(graph.name) + " Diameter: " + str(nx.diameter(graph)) + '\n')
    else:
        print(str(graph.name) + " Diameter: " + str(nx.diameter(graph)) + '\n')

    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.title(str(graph.name))
    plt.xlabel('Value')
    plt.xscale('log')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.scatter(deg, cnt, c='b', marker='x')
    plt.savefig(graph.name + '.png')
    plt.show()


def open_file_and_parse():
    with open('datasets/power.gml') as file:
        gml = file.read()
    gml = gml.split('\n')[1:]
    file.close()
    graph = nx.parse_gml(gml, label='id')
    return graph


def cluster_coefficient(graph):
    cluster = nx.average_clustering(graph)
    print(str(graph.name) + " Clustering Coefficient: " + str(cluster) + '\n')


def cluster(graph):
    partition = cl.best_partition(graph)
    pos = community_layout(graph, partition)

    options = {
        'node_color': list(partition.values()),
        'node_size': 10,
        'line_color': 'black',
        'linewidths': 1.5,
        'width': 1,
    }

    # nx.draw_networkx(graph, pos, node_color=list(partition.values()))
    plt.figure(figsize=(18, 18))
    plt.title(str(graph.name))
    nx.draw_networkx(graph, pos, node_size=100, node_color=list(partition.values()), nodelist=partition.keys(),
                     width=.5, with_labels=False)
    plt.savefig(graph.name + 'clustering' + '.png', format='PNG')
    plt.show()
    return


def community_layout(g, partition):
    """
    Compute the layout for a modular graph.


    Arguments:
    ----------
    g -- networkx.Graph or networkx.DiGraph instance
        graph to plot

    partition -- dict mapping int node -> int community
        graph partitions


    Returns:
    --------
    pos -- dict mapping int node -> (float x, float y)
        node positions

    """

    pos_communities = _position_communities(g, partition, scale=3.)

    pos_nodes = _position_nodes(g, partition, scale=1.)

    # combine positions
    pos = dict()
    for node in g.nodes():
        pos[node] = pos_communities[node] + pos_nodes[node]

    return pos


def _position_communities(g, partition, **kwargs):
    # create a weighted graph, in which each node corresponds to a community,
    # and each edge weight to the number of edges between communities
    between_community_edges = _find_between_community_edges(g, partition)

    communities = set(partition.values())
    hypergraph = nx.DiGraph()
    hypergraph.add_nodes_from(communities)
    for (ci, cj), edges in between_community_edges.items():
        hypergraph.add_edge(ci, cj, weight=len(edges))

    # find layout for communities
    pos_communities = nx.spring_layout(hypergraph, **kwargs)

    # set node positions to position of community
    pos = dict()
    for node, community in partition.items():
        pos[node] = pos_communities[community]

    return pos


def _find_between_community_edges(g, partition):
    edges = dict()

    for (ni, nj) in g.edges():
        ci = partition[ni]
        cj = partition[nj]

        if ci != cj:
            try:
                edges[(ci, cj)] += [(ni, nj)]
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    return edges


def _position_nodes(g, partition, **kwargs):
    """
    Positions nodes within communities.
    """

    communities = dict()
    for node, community in partition.items():
        try:
            communities[community] += [node]
        except KeyError:
            communities[community] = [node]

    pos = dict()
    for ci, nodes in communities.items():
        subgraph = g.subgraph(nodes)
        pos_subgraph = nx.spring_layout(subgraph, **kwargs)
        pos.update(pos_subgraph)

    return pos


def find_cluster_table(graph):
    table = defaultdict(int)
    partition = cl.best_partition(graph)
    for node in partition:
        val = partition[node]
        table[val] += 1

    name = str(graph.name) + "Cluster_Table.txt"
    file = open(name, 'w')
    for key, value in table.items():
        file.write(str(key) + "&" + str(value) + "\\\\" + "\n")
    file.close()


def betweenness_centrality(graph):
    bet_cen = nx.betweenness_centrality(graph)
    bet_sequence = sorted(bet_cen.values(), reverse=True)
    bet_count = collections.Counter(bet_sequence)
    bet, cnt = zip(*bet_count.items())
    plt.figure(figsize=(18, 18))
    plt.title(str(graph.name) + " Betweenness Centrality")
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.scatter(bet, cnt)
    plt.savefig(graph.name + 'between' + '.png')
    plt.show()


def real_calculations(graph):
    graph.degree()
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    print("Real Network Max Degree: " + str(max(degree_sequence)))
    bet_cen = nx.betweenness_centrality(graph)
    # bet_sequence = sorted(bet_cen.values(), reverse=True)
    print("Real Network Max Betweeness Centrality: " + "Node: " + str(
        max(bet_cen.items(), key=operator.itemgetter(1))[0]) + "Value: " + str(
        max(bet_cen.items(), key=operator.itemgetter(1))[1]))
    close_cen = nx.closeness_centrality(graph)
    # close_sequence = sorted(close_cen.values(), reverse=True)
    print("Real Network Max Closeness Centrality: " + "Node: " + str(
        max(close_cen.items(), key=operator.itemgetter(1))[0]) + "Value: " + str(
        max(close_cen.items(), key=operator.itemgetter(1))[1]))


def find_node(graph):
    degrees = list(graph.degree())
    print(max(degrees, key=operator.itemgetter(1))[0])
    print(max(degrees, key=operator.itemgetter(1))[1])


if __name__ == '__main__':
    main()
