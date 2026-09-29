from torch.utils.data import DataLoader

# Import project-specific artifact/provider modules before consulting registries
# when applications define additional BaseArtifact or ProviderBase subclasses.

from .engine.dynamic_collator import DynamicCollator
from .engine.multimodal_dataset import MultimodalDataset
from .engine.multi_dataset_manager import MultiDatasetManager

from .engine.layout.dataset_iteration_policy import DatasetIterationPolicy
from runweaver_ml.dataset_management.artifacts import create_artifacts
from .source import create_providers



def requested_modalities(
        cfg_group: dict,
) -> list[str]:
    out = list(cfg_group.get("requires", []))

    return out

def build_datasets(
       cfg_group:dict,
       runtime_artifact_repository=None,
    ):

    domain_filter = cfg_group.get("domain_filter", None)
    max_samples = cfg_group.get("max_samples", None)

    policy= DatasetIterationPolicy.from_config(
            cfg_group.get(
                "iteration_policy",
                {})
    )

    datasets = []
    for d_cfg in cfg_group.get("datasets", []):
        dataset_requires = list(cfg_group.get("requires", []))

        dataset = build_dataset(
                root= d_cfg["root"],
                required_modalities=dataset_requires,
                load_policy= cfg_group.get("lazy",[]),
                op_params= cfg_group.get("op_params", {}),
                domain=d_cfg["name"],
                bias=d_cfg.get("bias", 1.0),
                prefetch_workers=d_cfg.get("prefetch_workers", 1),
                runtime_artifact_repository=runtime_artifact_repository,
                artifacts=create_artifacts(
                    cfg_group.get("artifacts", []),
                    d_cfg,
                )

            )

        for provider in create_providers(cfg_group.get("providers", [])):
            dataset.install_provider(provider)

        datasets.append(dataset)

    if len(datasets) == 0:
        return None

    # setup conditional arguments to preserve
    # defaults from constructor
    manager_kwargs = {
        "max_samples": max_samples,
        "iteration_policy": policy
    }

    manager = MultiDatasetManager(
        datasets,
        **manager_kwargs,
    )
    if domain_filter is not None:
        manager.set_active_datasets(domain_filter)

    return manager


def build_dataset(
        root:str,                  # src directory containing modality folders
        required_modalities: list,  # or []
        load_policy:list,           # or[],
        op_params: dict,            # or {}
        domain: str,                # user friendly dataset name
        bias: float,            # range 0.0 - 1.0 relative sample  weights in multi dataset manager
        prefetch_workers: int = 1,
        artifacts=None,
        runtime_artifact_repository=None,
):

    dataset = MultimodalDataset(
        root= root,
        requires= required_modalities,
        load_policy= load_policy or[],
        op_params= op_params,
        domain= domain,
        bias= bias,
        prefetch_workers=prefetch_workers,
        artifacts=artifacts,
        runtime_artifact_repository=runtime_artifact_repository,

    )
    return dataset

# Parameter based connecting a loader and collator from existing datasets
def create_loader(dataset,
                 required_modalities:list,
                 batch_size:int
):


    modalities = list(required_modalities)
    for child_dataset in getattr(dataset, "datasets", [dataset]):
        modalities.extend(child_dataset.view.deliverables)

    collator = DynamicCollator(
        modalities=list(dict.fromkeys(modalities))
    )

    loader = DataLoader(
        dataset,
        batch_size= batch_size,
        shuffle=False,
        num_workers= 0, #num_workers,
        collate_fn=collator,
    )

    return loader

# complete config based loader and configured dataset(s)
def build_dataset_and_loader( cfg:dict):

    modalities = requested_modalities(cfg)

    return create_loader (
        build_datasets(
            cfg,
        ),
        modalities,
        cfg["batch_size"],
        #cfg["num_workers"]
    )
