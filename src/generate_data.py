import pandas as pd
import numpy as np

import random
import os

BASE_PATH = os.environ.get("HASOC_PATH")
DATA_PATHS_TRAIN = {
    "EN": f"{BASE_PATH}/data/raw/training_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/training_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/training_data/hindi_dataset.tsv"
}
DATA_PATHS_DEV = {
    "EN": f"{BASE_PATH}/data/raw/teaser_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/teaser_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/teaser_data/hindi_dataset.tsv"
}
DATA_PATHS_TEST = {
    "EN": f"{BASE_PATH}/data/raw/test_data/english_dataset.tsv",
    "DE": f"{BASE_PATH}/data/raw/test_data/german_dataset.tsv",
    "HI": f"{BASE_PATH}/data/raw/test_data/hindi_dataset.tsv"
}
print(DATA_PATHS_TRAIN)
DATA_COLUMNS = ["row_id", "text", "task_1", "task_2", "task_3"]
NUM_LANGUAGES = len(DATA_PATHS_TRAIN)


TASK_LABEL_IDS = {
    "task_1": ["NOT", "HOF"],
    "task_2": ["HATE", "OFFN", "PRFN"],
    "task_3": ["TIN", "UNT"],
    "task_4": [
        "NOT-NONE-NONE", 
        "HOF-HATE-TIN", "HOF-HATE-UNT", 
        "HOF-OFFN-TIN", "HOF-OFFN-UNT", 
        "HOF-PRFN-TIN", "HOF-PRFN-UNT", 
    ]
}

def generate_training_dev_data(args):
    for data_type, DATA_PATHS in [("train", DATA_PATHS_TRAIN), ("dev", DATA_PATHS_DEV), ("test", DATA_PATHS_TEST)]:
        print(data_type)
        for lang, path in DATA_PATHS.items():
            df = pd.read_csv(path, sep="\t").fillna("NULL")
            if data_type == "test":
                df = df.assign(**{
                    k: v[0]
                    for k,v in TASK_LABEL_IDS.items()
                })
            elif lang == "DE":
                df.loc[df.task_1 == "NOT", "task_3"] = "NONE"
                df.loc[df.task_1 != "NOT", "task_3"] = "TIN"
            
            # This is a fix for fixing errors in teasor data. Should not apply to test
            if data_type != "test":
                df.loc[df["task_1"] == "NOT", ["task_2", "task_3"]] = "NONE"
            
            if data_type != "test":
                for task in ["task_1", "task_2", "task_3"]:
                    df[task] = df[task].str.upper().replace("NULL", "NONE")
                df["task_4"] = df["task_1"].str.cat(df[["task_2", "task_3"]].astype(str), sep="-")
            task_cols = df.filter(regex=r'task_*', axis=1).columns
            for task in task_cols:
                if task == "task_3" and lang == "DE":
                    continue
                y = df[task]
                idx = (y != "NONE")
                df_t = df[idx]
                df_bert = pd.DataFrame({
                  'id':list(range(df_t.shape[0])),
                  'label':y[idx],
                  'alpha':['a']*df_t.shape[0],
                  'text': df_t["text"].replace(r'\s+', ' ', regex=True)
                })
                os.makedirs(os.path.join("./", lang, task), exist_ok=True)
                bert_format_path = os.path.join("./", lang, task, f"{data_type}.tsv")
                print(bert_format_path)
                df_bert.to_csv(bert_format_path, sep='\t', index=False, header=False)

def get_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    generate_training_dev_data(args)