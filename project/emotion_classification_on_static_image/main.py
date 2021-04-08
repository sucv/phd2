import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')

    # Server
    parser.add_argument('-experiment_name', default="static_image_classification", help='The experiment name.')
    parser.add_argument('-gpu', default=1, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance computing server or not?')
    parser.add_argument('-stamp', default='try_to_find_best_0407', type=str, help='To indicate different experiment instances')

    parser.add_argument('-model_load_path', default='/home/zhangsu/phd2/load', type=str, help='The path to load the trained model ')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', default='/home/zhangsu/phd2/save', type=str, help='The path to save the trained model ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', default='/home/zhangsu/phd2', type=str, help='The path to the entire repository.')  # /home/users/ntu/su012/phd2
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the model?')

    # Cross-validation
    parser.add_argument('-cross_validation', default=0, help='Use k-fold cross validation?')
    parser.add_argument('-num_folds', default=10, type=int, help='How many folds in total?')
    parser.add_argument('-folds_to_run', default=None, nargs="+", type=int, help='Which fold(s) to run in this session?')

    # Training
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')
    parser.add_argument('-dataset', default="affectnet", help='Dataset: ckplus, oulu, fer2013, ferp, rafd, rafdb, affectnet')
    parser.add_argument('-model_name', default="my_res50", help='The name to specify the model.')
    parser.add_argument('-model_mode', help='Mode: ir, ir_se', default="ir")
    parser.add_argument('-learning_rate', default=1e-3, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-num_epochs', default=2000, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-topk_accuracy', default=1, type=int, help='Whether the top k inferences covered the label?')
    parser.add_argument('-min_num_epochs', default=10, type=int, help='The minimum epoch to run at least.')

    # Scheduler
    parser.add_argument('-patience', default=15, type=int, help='The number of epoch to run before changing the learning rate.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-early_stopping', default=500, type=int, help='If no improvement, the number of epoch to run before halting the training')

    # Parameter Control
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', nargs="+", type=int, help='The specific epochs to do something.', default=[0])

    args = parser.parse_args()

    sys.path.insert(0, args.python_package_path)
    from project.emotion_classification_on_static_image.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
