config_mahnob = {
    "remote_root_directory": "/home/zhangsu/dataset/mahnob",
    "local_root_directory": r"E:\Mahnob_full",
    "raw_data_folder": "Sessions",
    "openface_output_folder": "processed_48",
    "npy_folder": "compacted_48",
    "emotion_dimension": ["Valence"],
    "modal": ["video", "eeg"],
    "metrics": ["rmse", "pcc", "ccc"],
    "num_classes": 3,
    "downsampling_interval_dict": {
        "frame": 16,
        "eeg": 64,
        "continuous_label": 1
    },
    "window_length": 24,
    "hop_size": 24,
    "continuous_label_frequency": 4,
    "frame_size": 48,
    "crop_size": 40,
    "batch_size": 3,
    "filename_pattern": {
        "continuous_label": "lable_continous_Mahnob.mat",
        "video": "P{}.+Section_{}.avi",
        "eeg": "Part_{}.+Trial{}.+bdf",
        "timestamp": "P{}.+Section_{}.tsv",
        "session_log": "session.xml"
    },
    "session_number": 239,
    "target_fps": 64,

    "openface_config": {
        "openface_directory": r"D:\OpenFace-master\x64\Release\FeatureExtraction",
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": "-aus",
        "output_image_flag": " -simalign ",
        "output_image_format": "-format_aligned jpg ",
        "output_image_size": " -simsize 48 ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir ",
        "output_image_mask_flag": " -nomask "
    }
}