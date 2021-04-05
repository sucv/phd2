import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-experiment_name', default="emotion_video", help='The experiment name.')
    parser.add_argument('-job', default=1, type=int, help='What job to run? 0: Regression Valence, 1: Classification Valence')
    parser.add_argument('-gpu', default=1, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, help='On high-performance server or not?')
    parser.add_argument('-stamp', default='eeg_cls_on_having_cont_only', type=str, help='To indicate different experiment instances')
    parser.add_argument('-dataset', default='mahnob_hci', type=str, help='The dataset name.')
    parser.add_argument('-modality', nargs="*", default=['eeg_image'])
    parser.add_argument('-resume', default=0, help='Resume from checkpoint?')

    parser.add_argument('-num_folds', default=10, type=int, help="How many folds to consider?")
    parser.add_argument('-folds_to_run', default=[1], nargs="+", type=int, help='Which fold(s) to run in this session?')

    parser.add_argument('-model_load_path', default='/home/zhangsu/phd2/load', type=str, help='The path to load the trained model.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', default='/home/zhangsu/phd2/save', type=str, help='The path to save the trained model ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', type=str, help='The path to the entire repository.', default='/home/zhangsu/phd2')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the model?')

    # Models
    parser.add_argument('-model_name', default="res_eeg", help='Model: res_eeg')
    parser.add_argument('-backbone_state_dict', default="state_dict_0.869", help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')
    parser.add_argument('-cnn1d_embedding_dim', default=512, type=int, help='Dimensions for temporal convolutional networks feature vectors.')
    parser.add_argument('-cnn1d_channels', default=[128, 128, 128, 128, 128], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-cnn1d_kernel_size', default=5, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-cnn1d_dropout', default=0.1, type=float, help='The dropout rate.')

    parser.add_argument('-lstm_embedding_dim', default=512, type=int, help='Dimensions for LSTM feature vectors.')
    parser.add_argument('-lstm_hidden_dim', default=256, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-lstm_dropout', default=0.4, type=float, help='The dropout rate.')

    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-num_epochs', default=1, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=0, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-time_delay', default=0, type=float, help='The time delay between input and label, in seconds.')
    parser.add_argument('-early_stopping', default=50, type=int, help='If no improvement, the number of epoch to run before halting the training')

    # Scheduler and Parameter Control
    parser.add_argument('-patience', default=10, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)
    from project.emotion_classification_on_mahnob_hci.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
