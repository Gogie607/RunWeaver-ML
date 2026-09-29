

#######################################
#
#    PhaseDataCompiler class
#  Rule and "contract Enforcer for  YAML configurations
#  Returns config as a dataclass
#
#######################################
"""
 description of the YAML file contents
 note about overrides:
        paramWrapper has a contextManager that is used to 
        automatically change any parameter that is defined in 'targets'
"""
PHASE_SCHEMA = {
    "name": "str",
    "enabled": "bool",

    "run": {
        "steps": "int",
        "trainable": ["list[str]"],
        "overrides": "dict | null"
    },

    "targets": {
        "<target_name>": "float"
    },

    "schedules": {
        "<target_name>": "schedule_config"
    }
}

class PhaseDataCompiler:

    # -----------------------------------------
    # RUNTIME COMPILATION(run once)
    # -----------------------------------------


    # -----------------------------------------
    # PHASE COMPILATION (per phase)
    # -----------------------------------------

    def compile_phase(self, phase_cfg):
        # --- REQUIRED FIELDS ---
        name = phase_cfg["name"]
        enabled = phase_cfg["enabled"]

        run = phase_cfg["run"]
        targets = phase_cfg["targets"]
        schedules = phase_cfg.get("schedules", {}) or {}

        # --- VALIDATION ---
        if not isinstance(run, dict):
            raise RuntimeError(f"[{name}] run must be dict")

        if not isinstance(targets, dict):
            raise RuntimeError(f"[{name}] targets must be dict")

        for k, v in targets.items():
            if not isinstance(v, (int, float, bool)):
                raise RuntimeError(f"[{name}] target '{k}' must be scalar or boolean")

        # --- FINAL OBJECT ---
        return {
            "name": name,
            "enabled": enabled,
            "run": run,
            "targets": {k: v for k, v in targets.items()},
            "schedules": schedules,
        }



#######################################
#
#  PhaseContext
#  container class containing all necessary data
#  from YAML to execute a single training/validation pass
#
#######################################
class PhaseContext:
    def __init__(
        self, *,
        name,
        enabled,
        run,
        targets,
        schedules,
        global_cfg,
        runtime_cfg,
        idx,
        logger
    ):
        # identity
        self.name = name
        self.enabled = enabled
        self.idx = idx

        # phase structure
        self.run = run
        self.targets = targets
        self.schedules = schedules

        # pass-through configs
        self.global_cfg = global_cfg
        self.runtime = runtime_cfg

        # logging
        self.logger = logger


    # ---- universally useful accessors ----

    @property
    def steps(self):
        return self.run["steps"]

    @property
    def trainable(self):
        return self.run["trainable"]

    def get(self, key, default=None):
        if key in self.targets:
            return self.targets[key]
        if key in self.run:
            return self.run[key]
        return default

    def require(self, key):
        if key in self.targets:
            return self.targets[key]
        if key in self.run:
            return self.run[key]
        raise KeyError(f"[{self.name}] missing required key: {key}")


    def __repr__(self):
        return (
            f"<PhaseContext {self.name} #{self.idx}\n"
            f"  enabled: {self.enabled}\n"
            f"  run: {self.run}\n"
            f"  targets: {self.targets}\n"
            f"  schedules: {self.schedules}\n"
            f">"
        )

#######################################
#
#   PhaseManager
#   Responsibilties:
#       configuration data verifications
#       context object creation
#       iteration and indexing
#
#######################################
class PhaseManager:
    def __init__(self, phases,  global_cfg, runtime_cfg, logger):
        self.contexts = []

        seen = set()
        compiler = PhaseDataCompiler()
        for idx, phase_cfg in enumerate(phases):
            compiled = compiler.compile_phase(phase_cfg)

            name = str(compiled["name"])

            if name in seen:
                raise RuntimeError(f"Duplicate phase name: {name}")
            seen.add(name)

            ctx = PhaseContext(
                name=name,
                enabled=compiled["enabled"],
                run=compiled["run"],
                targets=compiled["targets"],
                schedules=compiled["schedules"],

                # 👇 PASS-THROUGH ONLY
                global_cfg=global_cfg,
                runtime_cfg=runtime_cfg,

                idx=idx,
                logger=logger,
            )

            self.contexts.append(ctx)

    def __iter__(self):
        return iter(self.contexts)
#######################################
#
#
#######################################
