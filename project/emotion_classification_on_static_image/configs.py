img_size = 48
crop_size = 40

config_affectnet = {
    "remote_root_directory": "/media/affectnet/preprocessed",
    "local_root_directory": "E:\\AffectNet",
    "local_image_folder": "Manually_Annotated_Images",
    "local_label_filename_train": "training.csv",
    "local_label_filename_validate": "validation.csv",
    "local_output_directory": "E:\\AffectNet\\preprocessed",
    "output_image_size": 120,
    "landmark_number": 68,
    "resize": img_size,
    "center_crop": crop_size,
    "use_pretrained": True,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 8,
    "batch_size": 128
}

config_ckplus = {
    "local_root_directory": "E:\\CK+",
    "openface_output_folder": "E:\\CK+\\Cropped",
    "image_folder": "cohn-kanade-images",
    "label_folder": "Emotion",
    "remote_root_directory": "/home/zhangsu/dataset/CK+/Cropped",
    "resize": img_size,
    "center_crop": crop_size,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 8,
    "batch_size": 32,
    "use_pretrained": True,
    "openface_config": {
        "openface_directory": "D:\\OpenFace-master\\x64\\Release\\FeatureExtraction",
        "input_flag": " -fdir ",
        "output_features": " -2Dfp ",
        "output_action_unit": "",
        "output_image_flag": " -simalign ",
        "output_image_mask_flag": " -nomask ",
        "output_image_format": " -format_aligned jpg ",
        "output_image_size": " -simsize 120 ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    }
}

config_ferplus = {
    "remote_root_directory": "/home/zhangsu/dataset/fer+/preprocessed",
    "local_root_directory": "E:\\fer+",
    "root_csv_filename": "fer2013",
    "local_output_directory": "E:\\fer+\\preprocessed",
    "use_pretrained": True,
    "imbalanced": False,
    "resize": img_size,
    "center_crop": crop_size,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 8,
    "batch_size": 32
}

config_fer2013 = {
"remote_root_directory": "/home/zhangsu/dataset/fer2013/preprocessed",
    "local_root_directory": "E:\\fer2013",
    "root_csv_filename": "fer2013",
    "local_output_directory": "E:\\fer2013\\preprocessed",
    "resize": img_size,
    "center_crop": crop_size,
    "use_pretrained": True,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 7,
    "batch_size": 32
}

config_oulu = {
    "local_root_directory": "E:\\Oulu\\Original Video\\VL",
    "openface_output_folder": "E:\\Oulu\\Cropped",
    "remote_root_directory": "/home/zhangsu/dataset/Oulu/Cropped",
    "resize": img_size,
    "center_crop": crop_size,
    "use_pretrained": True,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 6,
    "batch_size": 32,
    "openface_config": {
        "openface_directory": "D:\\OpenFace-master\\x64\\Release\\FeatureExtraction",
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": "",
        "output_image_flag": " -simalign ",
        "output_image_mask_flag": " -nomask ",
        "output_image_format": " -format_aligned jpg ",
        "output_image_size": " -simsize 120 ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    }
}

config_rafd = {
    "local_root_directory": "E:\\rafd",
    "openface_output_folder": "E:\\rafd\\preprocessed",
    "remote_root_directory": "/home/zhangsu/dataset/rafd/Cropped",
    "resize": img_size,
    "center_crop": crop_size,
    "use_pretrained": True,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 8,
    "batch_size": 32,
    "openface_config": {
        "openface_directory": "D:\\OpenFace-master\\x64\\Release\\FeatureExtraction",
        "input_flag": " -f ",
        "output_features": " -2Dfp ",
        "output_action_unit": "",
        "output_image_flag": " -simalign ",
        "output_image_mask_flag": " -nomask ",
        "output_image_format": " -format_aligned jpg ",
        "output_image_size": " -simsize 120 ",
        "output_filename_flag": " -of ",
        "output_directory_flag": " -out_dir "
    }
}

config_rafdb = {
"remote_root_directory": "/home/zhangsu/dataset/rafdb/preprocessed",
    "local_root_directory": "E:\\rafdb",
    "local_image_folder": "Image\\aligned",
    "local_label_filename": "list_patition_label.txt",
    "local_output_directory": "E:\\rafdb\\preprocessed",
    "use_pretrained": True,
    "resize": img_size,
    "center_crop": crop_size,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
    "num_classes": 7,
    "batch_size": 32
}

