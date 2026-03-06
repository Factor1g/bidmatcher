
from .ml.build_traning_dataset import build_training_pipeline
from .ml.test_similarity import search
from .ml.predict_regression import predict_price

def main():
    print("BidMatch indul...")
    # ide jön a pipeline indítása

if __name__ == "__main__":
    # build_training_pipeline()
    # query = "Zenner vízóra | db"
    # results = search(query)

    # print("\nTop similar items:\n")
    # print(results.to_string(index=False))
    predict_price(
    "DN50 horganyzott acélcső szereléssel",
    "m"
    )
