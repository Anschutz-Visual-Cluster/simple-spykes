import json
import matplotlib.pyplot as plt


def graph_spikeinterface_quality_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        data = json.load(f)

    keys = list(data.keys())
    for k in keys:
        to_graph = data[k]
        x_vals = [int(v) for v in list(to_graph.keys())]
        y_vals = [v or 0 for v in list(to_graph.values())]

        plt.bar(x_vals, y_vals)
        plt.title(k)
        plt.show()
    tw = 2
