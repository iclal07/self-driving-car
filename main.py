from scripts.train_model import train
from scripts.test_model import test
from scripts.evaluate_model import evaluate

def main():
    # Modeli eğit
    train()

    # Modeli test et
    test()

    # Modeli değerlendir
    evaluate()

if __name__ == "__main__":
    main()

