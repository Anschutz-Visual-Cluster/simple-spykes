import traceback


class RawGraphData(object):
    def __init__(self):
        self.funcs = []
        self.values = {}

    def __getattr__(self, item):
        def func(*args, **kwargs):
            return self.add_func(func_name=item, args=args, kwargs=kwargs)
        return func

    def add_func(self, func_name, args=[], kwargs={}):
        tb = traceback.extract_stack()
        tb.pop()  # Exclude current frame in the stack
        self.funcs.append({"$func": func_name, "$args": args, "$kwargs": kwargs, "$traceback": tb})
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
