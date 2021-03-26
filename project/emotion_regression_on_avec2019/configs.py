frame_size = 48
crop_size = 40
window_length = 60
hop_size = 30

config_avec2019 = {
    "program_name": "avec2019",
    "metrics": ["rmse", "pcc", "ccc"],
    "emotion_dimension": ["Arousal", "Valence"],
    "frame_size": crop_size,
    "original_size": frame_size,
    "crop_size": crop_size,
    "batch_size": 2,
    "epochs": 50,
    "early_stopping": 25,
    "save_model": True,
    "save_metric_result": True,
    "window_length": window_length,
    "hop_size": hop_size,
    "remote_root_directory": "/home/zhangsu/dataset/avec2019",
    "local_root_directory": "E:\\AVEC2019",
    "raw_data_folder": "raw_data",
    "openface_output_folder": "processed_48",
    "npy_folder": "compacted_48",
    "partition_list": ["Train", "Devel", "Test"],
    "country_list": ["DE", "HU", "CN"],
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
        "output_image_size": " -simsize " + str(frame_size) + " ",
        "output_image_mask_flag": " -nomask ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    }
}