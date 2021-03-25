import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-exp', help='The experiment name.', default="img")
    parser.add_argument('-n_fold', type=int, help='How many folds in total?', default=10)
    parser.add_argument('-fold_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?', default=None)
    parser.add_argument('-d', help='Dataset: ckplus, oulu, fer2013, fer+, rafd, rafdb, affectnet', default="fer+")
    parser.add_argument('-m', help='Model: cfer, inceptresv1, my_inceptresv2, my_res50, my_vgg13, my_2dcnn', default="my_res50")
    parser.add_argument('-cv', help='Use k-fold cross validation?', default=False)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=1)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    parser.add_argument('-hpc', type=int, help='On high-performance computing server or not?', default=0)
    parser.add_argument('-s', type=str, help='To indicate different experiment instances', default='0001')
    parser.add_argument('-r', help='Resume from checkpoint?', default=False)
    parser.add_argument('-model_load_path', type=str, help='The path to load the trained model ',
                        default='/scratch/users/ntu/su012/pretrained_model')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', type=str, help='The path to save the trained model ',
                        default='/scratch/users/ntu/su012/trained_model')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', type=str, help='The path to the entire repository.',
                        default='/home/users/ntu/su012/phd2')  # /home/users/ntu/su012/phd2
    args = parser.parse_args()

    sys.path.insert(0, args.python_package_path)
    from project.emotion_classification_on_static_image.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
