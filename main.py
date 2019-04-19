import collections
import threading

import matplotlib.pyplot as plt
import networkx as nx


def main():
    non_random_graph = open_file_and_parse()
    non_random_graph.name = 'Real Graph'
    barabasi_graph = nx.barabasi_albert_graph(4941, 2)
    barabasi_graph.name = 'Barabasi Graph'
    random_graph = nx.erdos_renyi_graph(4941, 0.003, directed=False)
    random_graph.name = 'Random Graph'

    thread1 = threading.Thread(target=plot_degree_distribution, args=(non_random_graph,))
    thread2 = threading.Thread(target=plot_degree_distribution, args=(barabasi_graph,))
    thread3 = threading.Thread(target=plot_degree_distribution, args=(random_graph,))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()


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

    options = {
        'node_color': '#FF0000',
        'node_size': 10,
        'line_color': 'black',
        'linewidths': 1.5,
        'width': 1,
    }

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


if __name__ == '__main__':
    main()
