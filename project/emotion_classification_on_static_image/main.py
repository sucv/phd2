import argparse
from scripts.experiment_static_image_classification import generic_experiment_static_image_classification

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-n_fold', type=int, help='How many folds in total?', default=10)
    parser.add_argument('-fold_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?', default=None)
    parser.add_argument('-d', help='Dataset: ckplus, oulu, fer2013, fer+, rafd, rafdb, affectnet', default="fer+")
    parser.add_argument('-m', help='Model: cfer, inceptresv1, my_inceptresv2, my_res50, my_vgg13, my_2dcnn', default="my_res50")
    parser.add_argument('-cv', help='Use k-fold cross validation?', default=False)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=1)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    args = parser.parse_args()

    experiment_handler = generic_experiment_static_image_classification(args)
    experiment_handler.experiment()
