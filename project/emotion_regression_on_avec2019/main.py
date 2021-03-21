import sys

sys.path.insert(0, r'/home/zhangsu/phd2/')
from project.emotion_regression_on_avec2019.experiment import Experiment
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-exp', help='The project name.', default="avec2019")
    parser.add_argument('-tc', help='Subjects\' country for training set: DE, HU, all', default="all")
    parser.add_argument('-vc', help='Subjects\' country for validation set: DE, HU, all', default="all")
    parser.add_argument('-m', help='Model: 2d1d, 2dlstm', default="2d1d")
    parser.add_argument('-e', help='a: arousal, v: valence, b: both', default="b")
    parser.add_argument('-head', help='Multi-headed output? mh: multi-headed, sh: single-headed', default="mh")
    parser.add_argument('-lr', type=float, help='The initial learning rate.', default=1e-5)
    parser.add_argument('-d', type=float, help='Time delay between input and label, in seconds', default=0)
    parser.add_argument('-p', type=int, help='Patience for learning rate changes', default=4)
    parser.add_argument('-gpu', type=int, help='Which gpu to use?', default=1)
    parser.add_argument('-cpu', type=int, help='How many threads are allowed?', default=1)
    parser.add_argument('-s', type=str, help='To indicate different experiment instances', default='debug')
    parser.add_argument('-hpc', help='On high-performance computing server or not?', default=False)
    parser.add_argument('-model_load_path', type=str, help='The path to load the trained model ',
                        default='/home/zhangsu/phd2/load')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', type=str, help='The path to save the trained model ',
                        default='/home/zhangsu/phd2/save')  # /scratch/users/ntu/su012/trained_model
    args = parser.parse_args()

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
