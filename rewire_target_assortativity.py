import sys
import argparse
import random

import numpy as np
import networkx as nx
from tqdm import tqdm


def calculate_denominator(g):
    m = g.number_of_edges()
    sum_degrees = 0
    sum_degrees_squared = 0

    for u, v in g.edges():
        deg_u = g.degree[u]
        deg_v = g.degree[v]

        sum_degrees += deg_u + deg_v
        sum_degrees_squared += deg_u**2 + deg_v**2

    average_sum_degrees = sum_degrees/(2*m)
    average_sum_degrees_squared = sum_degrees_squared/(2*m)

    return average_sum_degrees_squared - (average_sum_degrees**2)

def calculate_numerator(g):
    m = g.number_of_edges()
    sum_product_degrees = 0
    sum_degrees = 0

    for u, v in g.edges():
        deg_u = g.degree[u]
        deg_v = g.degree[v]

        sum_product_degrees += deg_u * deg_v
        sum_degrees += deg_u + deg_v

    sum_product_degrees_normalized = sum_product_degrees/m
    average_sum_degrees = sum_degrees/(2*m)

    return sum_product_degrees_normalized - (average_sum_degrees**2)

def calculate_numerator_change(g, r, s, u, v):
    m = g.number_of_edges()
    # Current degrees
    k_r = g.degree[r]
    k_s = g.degree[s]
    k_u = g.degree[u]
    k_v = g.degree[v]

    product_before = k_r * k_s + k_u * k_v
    product_after = k_r * k_v + k_u * k_s
    return (product_after - product_before)/m

def degree_sequence_preserving_assortativity_shuffling(
        g, target_assortativity, step_rewires, b_factor, b0, min_acc):
    rewires = 0
    g = nx.Graph(g)
    denom = calculate_denominator(g)
    num = calculate_numerator(g)
    current_assortativity = num/denom

    b_step = b0
    acc = 1.0
    while abs(current_assortativity - target_assortativity) > 1e-9 \
            and acc > min_acc:
        accept = 0
        total = 0
        rewires = 0
        edges = list(g.edges())
        with tqdm(total=step_rewires, leave=False) as pbar:
            while rewires < step_rewires:
                e1, e2 = random.sample(edges, 2)

                r, s = e1
                u, v = e2

                if len(set([r, s, u, v])) < 4:
                    continue
                if g.has_edge(r, v) or g.has_edge(u, s):
                    continue

                new_num = num + calculate_numerator_change(g, r, s, u, v)

                new_assortativity = new_num/denom
                dist = abs(new_assortativity - target_assortativity) - \
                    abs(current_assortativity - target_assortativity)
                if random.random() < min(1, np.exp(-b_step * dist)):
                    g.remove_edge(r, s)
                    g.remove_edge(u, v)
                    g.add_edge(r, v)
                    g.add_edge(u, s)
                    edges = list(g.edges())

                    if dist > 1e-9:
                        accept += 1
                    num = new_num
                    current_assortativity = new_assortativity
                    rewires += 1
                    pbar.update(1)
                if dist > 1e-9:
                    total += 1

        # recalculate to avoid floating point drift
        num = calculate_numerator(g)
        acc = accept / total
        print(
            f"Assortativity: {current_assortativity}, "
            f"Target: {target_assortativity}, "
            f"Acceptance: {acc}, b_step: {1/b_step}",
            file=sys.stderr)
        b_step *= b_factor

    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input graph")
    parser.add_argument("target", type=float)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    args = parser.parse_args()
    g = nx.read_edgelist(args.input, comments="%")
    g2 = degree_sequence_preserving_assortativity_shuffling(
        g, target_assortativity=args.target,
        step_rewires=len(g.edges()),
        b_factor=1.7, b0=1, min_acc=1e-5)

    r2 = nx.degree_pearson_correlation_coefficient(g2)
    if abs(args.target - r2) >= args.epsilon:
        raise RuntimeError("did not converge")

    for i, j in g2.edges:
        print(i, j)
