import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def download_beir_dataset(dataset_name="scifact"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"Checking/Downloading {dataset_name} to {out_dir}...")
    data_path = util.download_and_unzip(url, out_dir)

    if dataset_name.lower() != "cqadupstack":
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        print(f"Successfully loaded {len(corpus)} documents and {len(queries)} queries.")
        return {dataset_name: (corpus, queries, qrels)}

    subsets = [
        "android", "english", "gaming", "gis", "mathematica", 
        "physics", "programmers", "stats", "tex", "unix", "webmasters", "wordpress"
    ]

    cqadupstack_data = {}
    for subset in subsets:
        subset_path = os.path.join(data_path, subset)
        if not os.path.isfile(os.path.join(subset_path, "corpus.jsonl")):
            raise FileNotFoundError(f"Missing corpus.jsonl for CQADupStack subset '{subset}': {subset_path}")
        
        print(f"Loading CQADupStack subset: {subset}")
        corpus, queries, qrels = GenericDataLoader(data_folder=subset_path).load(split="test")
        cqadupstack_data[subset] = (corpus, queries, qrels)

    print(f"Successfully loaded 12 independent CQADupStack subsets.")
    return cqadupstack_data

if __name__ == "__main__":
    download_beir_dataset("scifact")