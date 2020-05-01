from network import *

def main():
    trainer = None
    try:
        prepare_data_dir()
        trainer = CelebrityTrainer()
        trainer.train()
    finally:
        if trainer and trainer.confusion_matrix is not None:
            plot_confusion_matrix(trainer.confusion_matrix)


if __name__ == '__main__':
    main()
