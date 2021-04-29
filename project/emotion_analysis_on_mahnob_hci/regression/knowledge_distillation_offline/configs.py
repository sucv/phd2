config_knowledge_distillation = {
    "2d1d": {
        "model_name": "2d1d",
        "backbone_mode": "ir",
        "cnn1d_embedding_dim": 512,
        "cnn1d_channels": [128, 128, 128, 128],
        "cnn1d_kernel_size": 5,
        "cnn1d_dropout": 0.1,
        "teacher_frame_model_state_folder": "trained_2d1d_frame",
        "student_frame_backbone_state_folder": "backbone_frame",
        "teacher_eeg_image_model_state_folder": "trained_2d1d_eeg_image",
        "student_eeg_image_backbone_state_folder": "backbone_eeg_image"
    },
    "resnet": {
        "model_name": "resnet",
        "backbone_mode": "ir",
        "student_eeg_image_backbone_state_folder": "backbone_eeg_image"
    },
    "lstm": {
        "model_name": "2dlstm",
        "backbone_mode": "ir",
        "lstm_embedding_dim": 512,
        "lstm_hidden_dim": 256,
        "lstm_dropout": 0.4,
        "teacher_model_state_folder": "trained_2dlstm_frame"
    }
}