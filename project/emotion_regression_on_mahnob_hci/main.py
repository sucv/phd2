import sys

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-exp', help='The project name.', default="mahnob")
    parser.add_argument('-n_fold', type=int, help='What job to run? 0: Regression Valence', default=10)
    parser.add_argument('-folds_to_run', nargs="+", type=int, help='Which fold(s) to run in this session?', default=[0])
    parser.add_argument('-m', help='Model: 2d1d, 2dlstm', default="2d1d")
    parser.add_argument('-j', type=int, help='What job to run? 0: Regression Valence', default=0)
    parser.add_argument('-modal',  nargs="*", default=['frame', 'eeg'])
    parser.add_argument('-lr', type=float, help='The initial learning rate.', default=1e-5)
    parser.add_argument('-d', type=float, help='Time delay between input and label, in seconds', default=0)
    parser.add_argument('-p', type=int, help='Patience for learning rate changes', default=4)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=1)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    parser.add_argument('-s', type=str, help='To indicate different experiment instances', default='0754')
    parser.add_argument('-r', help='Resume from checkpoint?', default=0)
    parser.add_argument('-hpc', help='On high-performance computing server or not?', default=0)
    parser.add_argument('-model_load_path', type=str, help='The path to load the trained model ',
                        default='/home/zhangsu/phd2/load')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', type=str, help='The path to save the trained model ',
                        default='/home/zhangsu/phd2/save')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', type=str, help='The path to load the trained model ',
                        default='/home/zhangsu/phd2')  # /scratch/users/ntu/su012/pretrained_model

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)
    from project.emotion_regression_on_mahnob_hci.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
