import traceback


class RawGraphNone(object):
    pass


class RawGraphVariable(object):
    def __init__(self, name, default=RawGraphNone()):
        self.name = name
        self.default = default

    def __str__(self):
        return f"RawGraphVariable(\"{self.name}\")"


class RawGraphData(object):
    def __init__(self):
        self.funcs = []
        self.values = {}

    def __str__(self):
        return f"RawGraphData(<{len(self.funcs)}> funcs, <{len(self.values.keys())} vals>)"

    def __getattr__(self, item):
        def func(*args, **kwargs):
            return self.add_func(func_name=item, args=args, kwargs=kwargs)
        return func

    def add_func(self, func_name, args=[], kwargs={}):
        tb = traceback.extract_stack()
        tb.pop()  # Exclude current frame in the stack
        self.funcs.append({"$func": func_name, "$args": args, "$kwargs": kwargs, "$traceback": tb})
        return self

    def append_graph(self, other: 'RawGraphData'):
        for f in other.funcs:
            self.funcs.append(f)
        self.values.update(other.values)

    def replace_val_with_var(self, val):
        if isinstance(val, (list, tuple)):
            new_list = []
            for v in val:
                if isinstance(v, RawGraphVariable):
                    v = self.get_var(v)
                new_list.append(v)
            return new_list
        elif isinstance(val, RawGraphVariable):
            return self.get_var(val)
        elif isinstance(val, dict):
            new_dict = {}
            for k, v in val.items():
                if isinstance(v, RawGraphVariable):
                    v = self.get_var(v)
                new_dict[k] = v
            return new_dict
        raise ValueError(f"Unable to replace values in unknown type {val}!")

    def replace_vars(self):
        """
        Replace all variables with existing values

        :return: self
        """
        for func in self.funcs:
            func["$args"] = self.replace_val_with_var(func["$args"])
            func["$kwargs"] = self.replace_val_with_var(func["$kwargs"])

    def set_value(self, value_name, value):
        self.values[f"__{value_name}"] = value
        return self

    def get_var(self, var: RawGraphVariable):
        v = self.values.get(f"__{var.name}", RawGraphNone())
        if isinstance(v, RawGraphNone):
            if not isinstance(var.default, RawGraphNone):
                return var.default
            else:
                raise KeyError(f"Variable '{var.name}' not set and has no default!")
        return v

    def get_value(self, value_name):
        return self.values[f"__{value_name}"]

    def simple(self, func_name):
        self.add_func(func_name, {})
        return self

    def clf(self):
        return self.simple("clf")

    def show(self):
        return self.simple("show")
