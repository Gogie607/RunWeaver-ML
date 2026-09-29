
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
        return (
            (self.registry.has(key)) or
            key in self.run or
            key in self.global_cfg or
            key in self.runtime
        )

    def get(self, key):
        # dynamic (scheduled)
        if self.registry.has(key):
            return self.registry.get(key)

        # static layers
        for src in (self.run, self.global_cfg, self.runtime):
            if key in src:
                return src[key]
        # one-level nested lookup (your runtime case)
        for v in src.values():
            if isinstance(v, dict) and key in v:
                return v[key]
        return None