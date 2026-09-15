import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

def download_beir_dataset(dataset_name="scifact"):
    # Resolves deterministically to backend/data/datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "data", "datasets")
    os.makedirs(out_dir, exist_ok=True)
    
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    print(f"Checking/Downloading {dataset_name} to {out_dir}...")
    data_path = util.download_and_unzip(url, out_dir)
    
    # Load corpus (documents), queries, and official qrels (relevance judgments)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    print(f"Successfully loaded {len(corpus)} documents and {len(queries)} queries.")
    return corpus, queries, qrels

if __name__ == "__main__":
    download_beir_dataset("scifact")