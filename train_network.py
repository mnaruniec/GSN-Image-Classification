from network import *

def main():
    prepare_data_dir()
    trainer = CelebrityTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
