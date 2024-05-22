# Initialisation and topology effects in decentralised federated learning system

For running a simulation, generate a network in form of an edge file, using,
e.g., Reticula or NetworkX, and pass it to federated.py. To see all parameters and
options, simply run `python federated.py --help`.

```python
import networkx as nx
nx.write_edgelist(nx.erdos_renyi_graph(100, 0.1), "er_network.edgelist")
```

```sh
$ python federated.py er_network.edgelist --no-early-stopping --t-max 100 --items-per-user 128 --batches 8 --batch_size 16 --data-distribution balanced_iid --validation-split 0 --gain-correction sqrt --aggregation-method avg --test-exponential --test-base 1.1 --dataset mnist --arch simple-mnist --optimiser adam
```

The output is return in stdout as in JSON Lines (`jsonl`) format. Each communication round in one line.

The notebooks used for producing figures in the manuscript are presented in the `notebooks/` directory.
