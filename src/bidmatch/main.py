
from .ml.build_traning_dataset import build_training_pipeline
from .ml.test_similarity import search
from .eval.banchmark import create_test_set, create_testset ,run_full_benchmark
from .ml.run_matching import run_matching
from pathlib import Path

def main():
    print("BidMatch indul...")
    # ide jön a pipeline indítása

if __name__ == "__main__":
    
    #build_training_pipeline()
 
    # run_matching(
    #     input_path=Path("F:\Egyeb\Programozas\BidMatch\data\input\Kltgvetés_Kaba_árazatlan.xlsx"),
    #     output_path=Path("output_kaba.xlsx"),
    #     top_k=1,
    #     top_n_candidates=50,
    # )

    #create_test_set(n_samples=150)

    run_full_benchmark(
        testset_path=Path(r"F:\Egyeb\Programozas\BidMatch\data\test.csv"))

