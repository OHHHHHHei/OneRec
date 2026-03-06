import json
from tqdm import tqdm


def _normalize_cuda_list(cuda_list):
    if isinstance(cuda_list, int):
        return [str(cuda_list)]
    if isinstance(cuda_list, str):
        return [part.strip() for part in cuda_list.split(",") if part.strip()]
    return [str(part).strip() for part in cuda_list if str(part).strip()]


def merge(input_path, output_path, cuda_list):
    cuda_list = _normalize_cuda_list(cuda_list)
    if not cuda_list:
        raise ValueError("cuda_list must not be empty")
    data = []
    for i in tqdm(cuda_list):
        with open(f"{input_path}/{i}.json", "r") as f:
            data.extend(json.load(f))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
