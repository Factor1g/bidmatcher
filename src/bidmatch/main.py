
from .ml.build_traning_dataset import build_training_pipeline
from .ml.test_similarity import search
from .eval import banchmark
from .ml.run_matching import run_matching
from pathlib import Path

def main():
    print("BidMatch indul...")
    # ide jön a pipeline indítása



def main():
    print("BidMatch indul...")
    # ide jön a pipeline indítása

if __name__ == "__main__":
    
    build_training_pipeline()
 
    run_matching(
        input_path=Path("F:\Egyeb\Programozas\BidMatch\data\input\isb_test.xlsx"),
        output_path=Path("output.xlsx"),
        top_k=1,
        top_n_candidates=50,
    )
