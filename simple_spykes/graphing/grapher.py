from simple_spykes.graphing.raw_graphdata import RawGraphData
import matplotlib.pyplot as plt


class Grapher(object):
    def __init__(self, graph_data: RawGraphData, plotter):
        self.graph_data = graph_data
        self.plotter = plotter

    def _run_part(self, part: dict):
        func_name = part["$func"]
        params = part["$params"]
        tb = part["$traceback"]
        try:
            if isinstance(params, list):
                getattr(self.plotter, func_name)(*params)
            else:
                getattr(self.plotter, func_name)(**params)
        except Exception as e:

            raise ValueError(f"Error plotting Graph {part}!\n\nError: {str(e)}\n\nStack:\n\n {''.join(tb.format())}")

    def run(self):
        for part in self.graph_data.funcs:
            self._run_part(part)
        return self
