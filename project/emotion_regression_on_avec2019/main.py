import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('-experiment_name', default="avec2019_regression", help='The experiment name.')
    parser.add_argument('-gpu', default=1, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, help='On high-performance server or not?')
    parser.add_argument('-stamp', default='test', type=str, help='To indicate different experiment instances')
    parser.add_argument('-dataset', default='avec2019', type=str, help='The dataset name.')

    # Dataset
    parser.add_argument('-train_country', default="all", help='Subjects\' country for training set: DE, HU, all')
    parser.add_argument('-validate_country', default="all", help='Subjects\' country for validation set: DE, HU, all')

    # Path for Python code, model, datasets
    parser.add_argument('-dataset_load_path', default='/home/zhangsu/dataset/avec2019', type=str,
                        help='The root directory of the dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-dataset_folder', default='compacted_{:d}'.format(frame_size), type=str,
                        help='The root directory of the dataset.')  # /scratch/users/ntu/su012/dataset/mahnob
    parser.add_argument('-model_load_path', default='/home/zhangsu/phd2/load', type=str, help='The path to load the trained model.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-model_save_path', default='/home/zhangsu/phd2/save', type=str, help='The path to save the trained model ')  # /scratch/users/ntu/su012/trained_model
    parser.add_argument('-python_package_path', default='/home/zhangsu/phd2', type=str, help='The path to the entire repository.')  # /scratch/users/ntu/su012/pretrained_model
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the model?')

    parser.add_argument('-resume', default=1, type=int, help='Resume from checkpoint?')

    # Models
    parser.add_argument('-model_name', default="2d1d", help='Model: 2d1d, 2dlstm')
    parser.add_argument('-backbone_state_dict', default="model_state_dict_0.853", help='The filename for the backbone state dict.')
    parser.add_argument('-backbone_mode', default="ir", help='Mode for resnet50 backbone: ir, ir_se')
    parser.add_argument('-cnn1d_embedding_dim', default=512, type=int, help='Dimensions for temporal convolutional networks feature vectors.')
    parser.add_argument('-cnn1d_channels', default=[128, 128, 128, 128, 128], nargs="+", type=int, help='The size for each channel')
    parser.add_argument('-cnn1d_kernel_size', default=5, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-cnn1d_dropout', default=0.1, type=float, help='The dropout rate.')

    parser.add_argument('-lstm_embedding_dim', default=512, type=int, help='Dimensions for LSTM feature vectors.')
    parser.add_argument('-lstm_hidden_dim', default=256, type=int, help='The size of the 1D kernel for temporal convolutional networks.')
    parser.add_argument('-lstm_dropout', default=0.4, type=float, help='The dropout rate.')

    parser.add_argument('-train_emotion', default="b", help='The emotion dimension to focus when updating gradient, a: arousal, v: valence, b: both')
    parser.add_argument('-head', default="mh", help='Output 2 dimensions or 1? mh: multi-headed, sh: single-headed')
    parser.add_argument('-learning_rate', default=1e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-num_epochs', default=100, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=10, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-time_delay', default=0, type=float, help='The time delay between input and label, in seconds.')
    parser.add_argument('-early_stopping', default=20, type=int, help='If no improvement, the number of epoch to run before halting the training')

    # Groundtruth settings
    parser.add_argument('-num_classes', default=2, type=int, help='The number of classes for the dataset.')
    parser.add_argument('-emotion_dimension', default=["Arousal", "Valence"], nargs="*", help='The emotion dimension to analysis.')
    parser.add_argument('-metrics', default=["rmse", "pcc", "ccc"], nargs="*", help='The evaluation metrics.')

    # Dataloader settings
    parser.add_argument('-window_length', default=30, type=int, help='The length in second to windowing the data.')
    parser.add_argument('-hop_size', default=10, type=int, help='The step size or stride to move the window.')
    parser.add_argument('-continuous_label_frequency', default=10, type=int,
                        help='The frequency of the continuous label.')
    parser.add_argument('-frame_size', default=frame_size, type=int, help='The size of the images.')
    parser.add_argument('-crop_size', default=crop_size, type=int, help='The size to conduct the cropping.')
    parser.add_argument('-batch_size', default=2, type=int)

    # Scheduler and Parameter Control
    parser.add_argument('-patience', default=5, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=1, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')

    parser.add_argument('-save_plot', default=1, type=int, help='Whether to plot the session-wise output/target or not?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)
    from project.emotion_regression_on_avec2019.experiment import Experiment

    experiment_handler = Experiment(args)
    experiment_handler.experiment()
