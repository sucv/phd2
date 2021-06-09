import argparse
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')

    # Server
    parser.add_argument('-experiment_name', default="static_image_classification", help='The experiment name.')
    parser.add_argument('-gpu', default=1, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance computing server or not?')
    parser.add_argument('-stamp', default='bs16', type=str, help='To indicate different experiment instances') # pretrain_ResVIZ

    # Path for Python code, model, datasets
    parser.add_argument('-dataset_load_path', default='/home/zhangsu/dataset/affectnet/preprocessed', type=str,
                        help='The root directory of the dataset.')  # /scratch/users/ntu/su012/dataset/fer+/preprocessed
    parser.add_argument('-dataset_folder', default='', type=str, help='Useless for image experiment.')  # /scratch/users/ntu/su012/dataset/fer+/preprocessed
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
    parser.add_argument('-dataset', default="affectnet", help='Dataset: ckplus, oulu, fer2013, ferp, ferp_ce, rafd, rafdb, affectnet')
    parser.add_argument('-model_name', default="my_res50_ir", help='The name to specify the model.')
    parser.add_argument('-model_mode', default="ir", help='Mode: ir, ir_se')
    parser.add_argument('-learning_rate', default=1e-2, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-6, type=float, help='The minimum learning rate.')
    parser.add_argument('-num_epochs', default=125, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-topk_accuracy', default=1, type=int, help='Whether the top k inferences covered the label?')
    parser.add_argument('-min_num_epochs', default=10, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-use_weighted_sampler', default=1, type=int, help='Whether to balance the samples of each classes using weighted sampler?')
    parser.add_argument('-batch_size', default=32, type=int, help='The batch-size.')

    # Scheduler
    parser.add_argument('-patience', default=30, type=int, help='The number of epoch to run before changing the learning rate.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-early_stopping', default=7000, type=int, help='If no improvement, the number of epoch to run before halting the training')

    # Parameter Control
    parser.add_argument('-gradual_release', default=1, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=1, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[35, 65, 95], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=0, type=int,
                        help='Whether to load the best model state at the end of each epoch?')

    args = parser.parse_args()

    sys.path.insert(0, args.python_package_path)
    from project.emotion_classification_on_static_image.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
