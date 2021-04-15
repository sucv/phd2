frame_size = 48
crop_size = 40


config_avec2019 = {
    "program_name": "avec2019",
    "local_root_directory": "E:\\AVEC2019",
    "raw_data_folder": "raw_data",
    "openface_output_folder": "processed_{:d}".format(frame_size),
    "npy_folder": "compacted_{:d}".format(frame_size),
    "partition_list": ["Train", "Devel", "Test"],
    "country_list": ["DE", "HU", "CN"],
    "emotion_dimension": ["Arousal", "Valence"],
    "frame_size": frame_size,
    "crop_size": crop_size,
    "downsampling_interval_dict": {
        "frame": 5,
        "continuous_label": 1
    },
    "target_fps": 50,
    "openface_config": {
        "openface_directory": "D:\\OpenFace-master\\x64\\Release\\FeatureExtraction",
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": " -aus ",
        "output_image_flag": " -simalign ",
        "output_image_format": " -format_aligned jpg ",
        "output_image_size": " -simsize {:d} ".format(frame_size),
        "output_image_mask_flag": " -nomask ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    }
}