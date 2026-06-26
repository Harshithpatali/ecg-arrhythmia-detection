"""
Train CNN + BiLSTM + Attention model.
"""

from training.training.trainer import (
    ModelTrainer,
)


def main():

    trainer = ModelTrainer()

    trainer.train()


if __name__ == "__main__":

    main()