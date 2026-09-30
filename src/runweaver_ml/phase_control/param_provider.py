
# Parameter Provider class
# Provides a single truth source and abstraction from the yaml configuration structure

# Parameter definitions:
#  Run:
#       Non mutable values required by training/validation logic
#
# Global_cfg:
#       High level static data, usually common across test phases
#
# runtime:
#       administrative parameters used for train/val loop control (i,e logging, and model persistence)
#
# registry:
#        parameters that are allowed to be altered during a single training session

class ParamProvider:
    def __init__(self, *, run, global_cfg, runtime, registry):
        self.run = run
        self.global_cfg = global_cfg
        self.runtime = runtime
        self.registry = registry

    def has(self, key):
        found, _ = self._lookup(key)
        return found

    def get(self, key):
        found, value = self._lookup(key)
        return value if found else None

    def _lookup(self, key):
        # dynamic (scheduled)
        if self.registry.has(key):
            return True, self.registry.get(key)

        # static layers
        for src in (self.run, self.global_cfg, self.runtime):
            if key in src:
                return True, src[key]

        # one-level nested lookup, preserving layer precedence
        for src in (self.run, self.global_cfg, self.runtime):
            for value in src.values():
                if isinstance(value, dict) and key in value:
                    return True, value[key]

        return False, None
