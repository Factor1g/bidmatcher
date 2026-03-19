
from .ml.build_traning_dataset import build_training_pipeline
from .ml.test_similarity import search
from .ml.predict_regression import predict_price
from .ml.excel_price_suggester import main as excel_price_suggester_main
from .ml.excel_price_suggester import process_excel
from pathlib import Path

def main():
    print("BidMatch indul...")
    # ide jön a pipeline indítása

if __name__ == "__main__":
    
    #build_training_pipeline()
    process_excel(
        input_path=Path("F:\Egyeb\Programozas\BidMatch\data\input\isb_test.xlsx"),
        output_path=Path("output.xlsx")
    )
    
    #excel_price_suggester_main()
    # query = "Mitsubishi hőszivattyú"
    # unit = "db"
    # results = search(query, unit)

    # print("\nTop similar items:\n")
    # print(results.to_string(index=False))

    # predict_price(
    # "DN50 horganyzott acélcső szereléssel",
    # "m"
    # )
