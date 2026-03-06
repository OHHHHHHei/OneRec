import os
import pandas as pd


def _normalize_cuda_list(cuda_list):
    if isinstance(cuda_list, int):
        return [str(cuda_list)]
    if isinstance(cuda_list, str):
        return [part.strip() for part in cuda_list.split(",") if part.strip()]
    return [str(part).strip() for part in cuda_list if str(part).strip()]


def split(input_path, output_path, cuda_list):
    cuda_list = _normalize_cuda_list(cuda_list)
    if not cuda_list:
        raise ValueError("cuda_list must not be empty")
    df = pd.read_csv(input_path)
    # df = df.sample(frac=1).reset_index(drop=True)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    df_len = len(df)
    cuda_num = len(cuda_list)
    for i in range(cuda_num):
        start = i * df_len // cuda_num
        end = (i+1) * df_len // cuda_num
        df[start:end].to_csv(f'{output_path}/{cuda_list[i]}.csv', index=True)
