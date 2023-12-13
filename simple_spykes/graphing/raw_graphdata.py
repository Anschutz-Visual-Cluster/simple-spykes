import traceback


class RawGraphData(object):
    def __init__(self):
        self.funcs = []
        self.values = {}

    def add_func(self, func_name, param_dict, optional):
        tb = traceback.extract_stack()[:-1]  # Exclude this in the stack
        self.funcs.append({"$func": func_name, "$params": param_dict, "$traceback": tb})
        return self

    def add_value(self, value_name, value):
        self.values[f"__{value_name}"] = value
        return self

    def get_value(self, value_name):
        return self.values[f"__{value_name}"]

    def simple(self, func_name):
        self.add_func(func_name, {})
        return self

    def clf(self):
        return self.simple("clf")

    def show(self):
        return self.simple("show")
