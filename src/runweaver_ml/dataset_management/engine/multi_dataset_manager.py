import torch
from torch.utils.data import IterableDataset
from time import perf_counter

from .layout.dataset_iteration_policy import *
from runweaver_ml.dataset_management.artifacts import Artifact

class MultiDatasetManager(IterableDataset):
    def __init__(self,
                 datasets,
                 iteration_policy,
                 max_samples=None):
        self.datasets = datasets

        self.iteration_policy = iteration_policy or DatasetIterationPolicy()

        self.max_samples = max_samples

        self.active_datasets = datasets

        # Optional components
        # TODO  add weight scheduling
        self.scheduler = None

        self.weights = self._compute_weights(
            self.datasets
        )
        self.active_weights = self.weights

        self._wait_report = {}

    def _compute_weights(self, datasets):
        """
         Dataset sampling probabilities are derived from:

           (dataset_size ** alpha) * bias

         where:

           alpha = 1.0
               proportional-to-size sampling

           alpha = 0.0
               uniform dataset sampling

           0.0 < alpha < 1.0
               tempered balancing favoring larger datasets
               without allowing domination

         bias allows manual domain emphasis/suppression.
        """
        alpha = 0.5

        scaled = [
            (ds.num_samples ** alpha) * ds.bias
            for ds in datasets
        ]

        total = sum(scaled)

        return [
            w / total if total > 0 else 1
            for w in scaled
        ]

    def set_active_datasets(self, domain_filter= None):

        if domain_filter is None:
            self.active_datasets = self.datasets
            return

        self.active_datasets = []
        self.active_weights = []

        found = [
            ds for ds in self.datasets
            if ds.domain in domain_filter
        ]

        if not found:
            raise ValueError(
                f"No datasets matched: {domain_filter}"
            )

        missing = set(domain_filter) - {
            ds.domain for ds in found
        }

        if missing:
            print(
                f"[WARN] Unknown datasets ignored: "
                f"{sorted(missing)}"
            )

        self.active_datasets = found


        self.active_weights = self._compute_weights(
            self.active_datasets
        )
    #----------------------------------
    def attach_scheduler(self, scheduler):
        self.scheduler = scheduler

    #----------------------------------
    def __iter__(self):
        if self.iteration_policy.mix_mode is MixMode.MIXED:
            yield from self._iter_mixed()

        elif self.iteration_policy.mix_mode is MixMode.PER_DOMAIN:
            yield from self._iter_per_domain()

        else:
            raise RuntimeError(
                f"Unknown mix mode: {self.iteration_policy.mix_mode}"
            )

    #----------------------------------
    def _iter_per_domain(self):

        sample_count = 0

        for ds in list(self.active_datasets):

            #domain = getattr(ds, "domain", None)


            #if (
            #        domain not in self.domain_filter
            #):
            #    continue

            for item in ds:

                yield item

                sample_count += 1

                if (
                        self.max_samples is not None
                        and sample_count >= self.max_samples
                ):
                    return

    #----------------------------------
    def _iter_mixed(self):
        datasets =  list(self.active_datasets)
        weights = list(self.active_weights)

        iters = [
            iter(ds)
            for ds in datasets
        ]

        sample_count = 0

        # sampling probabilities for active datasets
        total = sum(weights)

        probs = torch.tensor(
            [w / total for w in weights],
            dtype=torch.float
        )

        while (
                iters
                and (
                        self.max_samples is None
                        or sample_count < self.max_samples
                )
        ):

            idx = torch.multinomial(
                probs,
                1
            ).item()

            try:
                wait_start = perf_counter()
                item = next(iters[idx])

                wait_s = perf_counter() - wait_start

                if wait_s >= 0.01:
                    domain = datasets[idx].domain

                    report = self._wait_report.setdefault(
                        domain,
                        {
                            "count": 0,
                            "total_s": 0.0,
                            "max_s": 0.0,
                        },
                    )

                    report["count"] += 1
                    report["total_s"] += wait_s
                    report["max_s"] = max(
                        report["max_s"],
                        wait_s,
                    )

                sample_count += 1

                yield item

            except StopIteration:

                if self.iteration_policy.exhaust_policy is ExhaustPolicy.LOOP:

                    iters[idx] = iter(datasets[idx])

                elif self.iteration_policy.exhaust_policy is ExhaustPolicy.FINITE_EXHAUSTIVE:

                    del iters[idx]
                    del datasets[idx]
                    del weights[idx]

                    if not iters:
                        break

                    # sampling probabilities for active datasets
                    total = sum(weights)

                    probs = torch.tensor(
                        [w / total for w in weights],
                        dtype=torch.float
                    )

                elif self.iteration_policy.exhaust_policy is ExhaustPolicy.FINITE_STATIONARY:

                    break

                else:

                    raise RuntimeError(
                        f"Unknown exhaust policy: "
                        f"{self.iteration_policy.exhaust_policy}"
                    )

    #----------------------------------
    def update_weights(self, metrics):
        if not self.scheduler:
            return

        new_weights = self.scheduler.get_weights(metrics)
        # Assign new weights (must match dataset names)
        for i, ds in enumerate(self.datasets):
            name = getattr(ds, 'domain', f"ds_{i}")
            self.weights[i] = new_weights.get(name, self.weights[i])

    #-----------------------------------------
    def distribute_artifact(
            self,
            name: str,
            value=None,
            create=None,
    ):
        """
        Distribute an artifact to all managed datasets.

        Parameters
        ----------
        name : str
            Artifact name.

        value : Any, optional
            Constant value stored directly.

        create : Callable[[MMWDS], Any], optional
            Dataset-aware bootstrap function.
            Called once per dataset:

                obj = create(ds)

            The returned object is stored.

        Note:  There are subtle differences between usage
            3  nearly similar examples 3 entirely different outcomes

                # Shared RNG instance
                value=random.Random(42)   0
                    # value stores the result of random(42)

                # Store RNG factory
                value=lambda ds: random.Random(42)
                    # value stores  shared  callable fcn reference 1 object

                # Create RNG instance per dataset
                create=lambda ds: random.Random(42)
                    # value stores  a new  callable fcn reference for each dataset

        """

        if (value is None) == (create is None):
            raise ValueError(
                "Specify exactly one of "
                "'value' or 'create'"
            )

        for ds in self.datasets:

            obj = (
                create(ds)
                if create is not None
                else value
            )

            ds.artifact_store.register(
                Artifact(
                    name=name,
                    value=obj,
                )
            )

    @staticmethod
    def _identity(x):
        return x

    def aggregate_artifacts(
            self,
            artifact_name,
            aggregate_fn= None
    ):
        items = []

        for ds in self.datasets:
            art = ds.artifact_store.get(
                artifact_name
            )

            items.append(
                art.item()
            )

        return (
            items
            if aggregate_fn is None
            else aggregate_fn(items)
        )

    def consume_wait_report(self):
        report = self._wait_report
        self._wait_report = {}
        return report