import argparse
import random
import numpy as np
import networkx as nx


def pick_edge_pairs(g, node_degree, degree_nodes):
    while True:
        i = random.choice(list(g.nodes()))
        j = random.choice(list(g.neighbors(i)))

        di = node_degree[i]
        if len(degree_nodes[di]) < 2:
            continue

        a = random.choice(degree_nodes[di])
        while a == i:
            a = random.choice(degree_nodes[di])

        b = random.choice(list(g.neighbors(a)))
        if a == j or b == i or g.has_edge(i, b) or g.has_edge(a, j):
            continue

        return (i, j), (a, b)


def pkk_preserving_rewire(g, max_rewires):
    g = nx.Graph(g)
    node_degree = {n: g.degree(n) for n in g.nodes}
    degree_nodes = {}
    for node, degree in node_degree.items():
        if degree not in degree_nodes:
            degree_nodes[degree] = []
        degree_nodes[degree].append(node)

    rewires = 0
    while rewires < max_rewires:
        (i, j), (a, b) = pick_edge_pairs(g, node_degree, degree_nodes)
        g.remove_edge(i, j)
        g.remove_edge(a, b)
        g.add_edge(i, b)
        g.add_edge(a, j)

        rewires += 1

        if rewires % 500 == 0:
            print(100*rewires/max_rewires)

    return g


def pkk_cbar_preserving_rewire(
        g, premix_rewires, step_rewires, b_factor, b0, min_acc):
    c_t = nx.average_clustering(g)
    g = pkk_preserving_rewire(g, premix_rewires)

    c = nx.average_clustering(g)
    node_degree = {n: g.degree(n) for n in g.nodes}
    degree_nodes = {}
    for node, degree in node_degree.items():
        if degree not in degree_nodes:
            degree_nodes[degree] = []
        degree_nodes[degree].append(node)

    norm = len(g.nodes)

    b_step = b0
    print(c, c_t, 1/b_step)
    acc = 1.0
    while abs(c - c_t) > 0 and acc > min_acc:
        accept = 0
        total = 0
        rewires = 0
        while rewires < step_rewires:
            (i, j), (a, b) = pick_edge_pairs(g, node_degree, degree_nodes)

            n_i = set(g.neighbors(i))
            n_j = set(g.neighbors(j))
            n_a = set(g.neighbors(a))
            n_b = set(g.neighbors(b))

            c_h = c
            for v in n_i & n_j:
                c_h -= 2/(node_degree[i]*(node_degree[i]-1))/norm
                c_h -= 2/(node_degree[j]*(node_degree[j]-1))/norm
                c_h -= 2/(node_degree[v]*(node_degree[v]-1))/norm
            for v in n_a & n_b:
                c_h -= 2/(node_degree[a]*(node_degree[a]-1))/norm
                c_h -= 2/(node_degree[b]*(node_degree[b]-1))/norm
                c_h -= 2/(node_degree[v]*(node_degree[v]-1))/norm

            for v in n_i & n_b:
                if v not in {j, a}:
                    c_h += 2/(node_degree[i]*(node_degree[i]-1))/norm
                    c_h += 2/(node_degree[b]*(node_degree[b]-1))/norm
                    c_h += 2/(node_degree[v]*(node_degree[v]-1))/norm
            for v in n_a & n_j:
                if v not in {b, i}:
                    c_h += 2/(node_degree[a]*(node_degree[a]-1))/norm
                    c_h += 2/(node_degree[j]*(node_degree[j]-1))/norm
                    c_h += 2/(node_degree[v]*(node_degree[v]-1))/norm

            dist = abs(c_h - c_t) - abs(c - c_t)

            if random.random() < min(1, np.exp(-b_step * dist)):
                g.remove_edge(i, j)
                g.remove_edge(a, b)
                g.add_edge(i, b)
                g.add_edge(a, j)

                if dist != 0:
                    accept += 1
                c = c_h
                rewires += 1

            if dist != 0:
                total += 1
        acc = accept/total
        print(c, c - c_t, 1/b_step, acc)
        b_step *= b_factor

    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input graph")
    args = parser.parse_args()
    g = nx.read_edgelist(args.input, comments="%")
    g2 = pkk_cbar_preserving_rewire(
            g, 10*len(g.edges()), 10*len(g.edges()),
            1.7, 1e-2, 1e-7)
    print("nodes:", len(g.nodes), len(g2.nodes))
    print("edges:", len(g.edges), len(g2.edges))
    print("degs", sorted(g.degree()) == sorted(g2.degree()))
    print("cbars", nx.average_clustering(g), nx.average_clustering(g2))
