import os

from minionerec.cli._shared import build_config
from minionerec.config.schema import ConvertConfig
from minionerec.data.convert import convert_interactions_to_csv, create_item_info_file, load_dataset


def run_convert_cli(config_path: str | None, overrides: list[str] | None):
    config = build_config(ConvertConfig, config_path, overrides)
    dataset = load_dataset(config.data.data_dir, config.data.dataset_name)
    category = config.data.category or config.data.dataset_name
    info_path = config.extras.get("info_path") or os.path.join(config.data.output_dir, "info", f"{category}_5_2016-10-2018-11.txt")
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    create_item_info_file(dataset["items"], dataset["item_to_semantic"], info_path)
    for split_name in ["train", "valid", "test"]:
        if split_name in dataset["splits"]:
            split_output_dir = os.path.join(config.data.output_dir, split_name)
            os.makedirs(split_output_dir, exist_ok=True)
            convert_interactions_to_csv(
                {split_name: dataset["splits"][split_name]},
                dataset["items"],
                dataset["item_to_semantic"],
                split_output_dir,
                category,
                seed=config.training.seed,
                keep_longest_only=bool(config.extras.get("keep_longest_only", False)),
            )
    return config.data.output_dir
