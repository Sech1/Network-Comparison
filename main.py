import collections

import matplotlib.pyplot as plt
import networkx as nx


def main():
    non_random_graph = open_file_and_parse()
    non_random_graph.name = 'non_random_graph'
    barabasi_graph = nx.barabasi_albert_graph(1589, 2)
    barabasi_graph.name = 'barabasi_graph'
    random_graph = nx.erdos_renyi_graph(1589, 0.00259)
    random_graph.name = 'random_graph'

    plot_degree_distribution(non_random_graph)
    plot_degree_distribution(barabasi_graph)
    plot_degree_distribution(random_graph)


def plot_degree_distribution(graph):
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)
    degree_count = collections.Counter(degree_sequence)
    deg, cnt = zip(*degree_count.items())
    print(graph.number_of_edges())

    options = {
        'node_color': '#FF0000',
        'node_size': 10,
        'line_color': 'black',
        'linewidths': 1.5,
        'width': 1,
    }

    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(deg, cnt, c='b', marker='x')
    plt.savefig(graph.name + '.png')
    plt.show()


def open_file_and_parse():
    with open('datasets/netscience.gml') as file:
        gml = file.read()
    gml = gml.split('\n')[1:]
    file.close()
    graph = nx.parse_gml(gml)
    return graph


if __name__ == '__main__':
    main()
