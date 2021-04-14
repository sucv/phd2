# The configs here are for preprocessing.  They are set according to the data collection settings of the dataset.
# They are supposed to be fixed except for frame_size and crop_size.

frame_size = 48

config_mahnob = {
    "local_root_directory": r"E:\Mahnob_full",  # Local root for data preprocessing.
    "raw_data_folder": "Sessions",
    "openface_output_folder": "processed_{:d}".format(frame_size),
    "npy_folder": "compacted_{:d}".format(frame_size),
    "downsampling_interval_dict": {
        "frame": 16,
        "eeg": 64,
        "continuous_label": 1
    },
    "frame_size": frame_size,
    "filename_pattern": {
        "continuous_label": "lable_continous_Mahnob.mat",
        "video": "P{}.+Section_{}.avi",
        "eeg": "Part_{}.+Trial{}.+bdf",
        "timestamp": "P{}.+Section_{}.tsv",
        "session_log": "session.xml"
    },
    "target_fps": 64,
    "openface_config": {
        "openface_directory": r"D:\OpenFace-master\x64\Release\FeatureExtraction",
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": "-aus",
        "output_image_flag": " -simalign ",
        "output_image_format": "-format_aligned jpg ",
        "output_image_size": " -simsize {:d} ".format(frame_size),
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir ",
        "output_image_mask_flag": " -nomask "
    }
}
