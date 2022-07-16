from imports import *
from args import *
from utils import *
from data import *
from networks import *
from trainer import *

if __name__ == "__main__":
    args = TrainingArgs()
    data, weights = get_data(args)

    args.data = data
    args.weights = weights

    trainer = Trainer(args)

    