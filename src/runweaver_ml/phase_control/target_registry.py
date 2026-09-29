# Target params registry and abstraction



class TargetRegistry:
    def __init__(self, initial_targets: dict):
        self._values = {}

        for k, v in initial_targets.items():
            self._values[k] = v

    def has(self, name):
        return name in self._values

    def get(self, name):
        if name not in self._values:
            raise KeyError(f"Target '{name}' not registered")
        return self._values[name]

    def set(self, name, value):
        if name not in self._values:
            raise KeyError(f"Target '{name}' not registered")
        self._values[name] = value

    def as_dict(self):
        return dict(self._values)
