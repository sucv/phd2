

# Table of contents <a name="Table_of_Content"></a>

+ [Preprocessing](#Preprocessing)
    + [Environment](#PE) 
    + [Feature Extraction and Data Preparation](#FEDP)
+ [Cross-modal Knowledge Distillation](#CKD)
    + [Environment](#TE)
    + [Step One, Resnet Backbone Fine-tuning](#S1)
    + [Step Two, Teacher Model Training](#S2)
    + [Step Three, Knowledge Extraction](#S3)
    + [Step Four, Knowledge Distillation](#S4)


## Preprocessing<a name="Preprocessing"></a>
[Return to Table of Content](#Table_of_Content)

The preprocessing is meant to be done in Windows system, with an IDE like PyCharm. The preprocessing do not use GPU so any PC should be okay. 

If you are using Linux system, then you may have to  compile the OpenFace toolbox from source code. 

(To code examiner: To make life easier if you are using Linux, alternatively, you may choose to download the processed
  files. Since the files are existing, the command for OpenFace toolbox will not be executed because I largely used:

```
If file not exist:
    then run something
``` 

(The processed folder is small in size, and downloading them will not influence the debugging @_@)

#### Environment<a name="PE"></a>
[Return to Table of Content](#Table_of_Content)

First, establish a anaconda virtual environment by executing the following commands. (Then, you will need to set the Python interpreter accordingly for
 your PyCharm.)

```
conda create -n zs_pre python==3.8
conda activate zs_pre

conda install -c anaconda pandas
pip install opencv-python
conda install -c anaconda tqdm 
conda install -c anaconda scipy 
conda install -c anaconda scikit-learn
conda install -c conda-forge mne
pip install mne
conda install -c anaconda pillow
pip install eeg_positions
conda install -c conda-forge ffmpeg
```

Second, install the newest OpenFace (2.2.0 so far) from [this link](https://github.com/TadasBaltrusaitis/OpenFace/releases). For Windows users, just directly download and unzip 
compiled files. For Linux users, you will have to compile using the source code. 

Third, two paths need to be set. 

1. In `project/emotion_analysis_on_mahnob_hci/configs.py`:
    + Set `local_root_directory` to the dataset root path.
    + (If you use OpenFace for preprocessing) Set `openface_directory` to the absolute path of the OpenFace file named "FeatureExtraction.exe".
    
Now you are good to go.

#### Feature Extraction and Data Preparation<a name="FEDP"></a>
[Return to Table of Content](#Table_of_Content)

Run `project/emotion_analysis_on_mahnob_hci/preprocessing.py` using Pycharm, everything shall be done. The code mainly produce the
 cropped and aligned facial images, and their corresponding EEG PSD features and continuous labels for each trial.
 

## Cross-modal Knowledge Distillation<a name="CKD"></a>
[Return to Table of Content](#Table_of_Content)

Given the knowledge (features) from the visual modality, the input (PSD) from the EEG modality, and 
the continuous label of valence, the goal of this project is to improve the consistency of the model predictions and the labels. 
 It's an N-to-N task, i.e., given a window containing N samples, the output shall be N predictions. The consistency is evaluated using Pearson Correlation Coefficient (PCC) and 
 Concordance Correlation Coefficient (CCC).
 
 Two scenarios in regards of data partitioning are involved.
 
 1. Trial-level shuffling, which firstly shuffles all the trials, and then separates them into 10 folds evenly. Trials of the same subject
  may appear in different partitions.
 2. Leave-one-subject-out, which guarantees that the test set has strictly trials from one subject. The training and validation sets are randomly chosen 
 following a ratio of 8 to 2 from the remaining trials.
 
 (To examiner: The above two scenarios are the focus for code examination. Theoretically, the first scenario shall obtain better result than the second one. Yet the actual
  result manifests oppositely on PCC. We see below that the PCC of Scenario 2 is higher than that from Scenario 1. Why?)
  
  | Scenario 1        | PCC           | CCC  |
| ------------- |:-------------:| -----:|
| w/o KD     | 0.400 | 0.378 |
| w/  KD    | 0.412      |   0.387 |
| P-value | 0.01      |    0.034 |

  | Scenario 2        | PCC           | CCC  |
| ------------- |:-------------:| -----:|
| w/o  KD    | 0.464 | 0.363 |
| w/   KD   | 0.474      |   0.372 |
| P-value | 0.007      |    0.047 |

This part of tasks can be run on command line or Pycharm.

#### Environment<a name="TE"></a>
[Return to Table of Content](#Table_of_Content)

```
conda create -n zs python==3.8
conda activate zs

conda install -c anaconda pandas
pip install opencv
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c anaconda scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib

```

#### Step One, Resnet Backbone Fine-tuning<a name="S1"></a>  
[Return to Table of Content](#Table_of_Content)

The goal is to train a good Resnet50 for video frame encoding. The code
  to do this is located in `project/emotion_classification_on_static_image/main.py`.
   
The Resnet50 is downloaded from [this link](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#model-zoo).
 The model is trained using MS-Celeb-1M_align dataset as a facial recognition task.
 
 The model is then fine-tuned using Fer+ dataset as a facial expression classification task. 
  
 Instead of train it again, you may download the trained backbone from [this link](https://drive.google.com/file/d/1egC4dG3FhuDRBRI1xhQrzkIRQ0vFSxBX/view?usp=sharing).
  
  (To examiner: for debugging purpose, it is okay to omit this step and using the trained backbone directly.)
  
#### Step Two, Teacher Model Training<a name="S2"></a>
[Return to Table of Content](#Table_of_Content)

The goal is to train the teacher model using video frame, and save the model parameters for knowledge (feature) extraction.

The code to do so is located in `project/emotion_analysis_on_mahnob_hci/regression/main.py`.

First, please specify `dataset_load_path`, `model_load_path`, `model_save_path`, `python_package_path` accordingly. The Resnet50 backbone should be placed in `model_load_path`.

(To examiner: the `debug` is already set to 1 for your convenience. In this case, only 6 trials from 3 subjects are loaded, and the epoch number is set to 1.)

Second, execute `project/emotion_analysis_on_mahnob_hci/regression/main.py` using the following two settings, one at a time:

1. Set `case` to `trial` for trial-level shuffling scenario.
2. Set `case` to `loso` for a standard Leave-one-subject-out scenario.

Third, move the trained model to specified directory accordingly. For each case we trained, we have three folds, namely three models.

(To examiner: the trained and arranged teacher model folder is provided privately. They trained models are only for debugging purpose, since 
they are copied and pasted from the same file.).

1. For `trial` case, copy and rename the trained model (i.e., `model_state_dict.pth`) from fold 0, 1, 2, ...,  obtaining:
    1. `teacher_model_folder/model_trial/2d1d_0.pth`
    1. `teacher_model_folder/model_trial/2d1d_1.pth`
    1. `teacher_model_folder/model_trial/2d1d_2.pth`
    1. ...

1. For `loso` case, copy and rename the trained model (i.e., `model_state_dict.pth`) from fold 0, 1, 2, ...,  obtaining:
    1. `teacher_model_folder/model_loso/2d1d_0.pth`
    1. `teacher_model_folder/model_loso/2d1d_1.pth`
    1. `teacher_model_folder/model_loso/2d1d_2.pth`
    1. ...

Note, the folder `teacher_model_folder` has to be placed in your `model_load_path`.

#### Step Three, Knowledge Extraction<a name="S3"></a>
[Return to Table of Content](#Table_of_Content)

The goal is to use the trained teacher to extract knowledge.  
The model will be run on the training data again to extract temporal features, i.e., the output of the TCN or LSTM network, before the regressor.

A whole video will be fed into the trained teacher model, producing the features of a trial in one go. It may consume over 10G graphic memory.

The code to do so is located in :

    + `project/emotion_analysis_on_mahnob_hci/regression/knowledge_distillation_offline/extract_knowledge_trial`
    + `project/emotion_analysis_on_mahnob_hci/regression/knowledge_distillation_offline/extract_knowledge_LOSO`
    

First, please specify `dataset_load_path`, `model_load_path`, `model_save_path`, `python_package_path` accordingly. Note, the folder `teacher_model_folder` has to be placed in your `model_load_path`.

Second, run the two code files for trial-wise shuffling and LOSO scenarios, repsectively. They will generate two folders containing 
 the extracted knowledge (feature):
    
    + `knowledge_folder/knowledge_2d1d_frame_trial/...`,
    + `knowledge_folder/knowledge_2d1d_frame_loso/...`.
    
#### Step Four, Knowledge Distillation<a name="S4"></a>
[Return to Table of Content](#Table_of_Content)

The goal is to teach EEG psd model using visual knowledge obtained above.

The code to do so is located in `project/emotion_analysis_on_mahnob_hci/regression/knowledge_distillation/main_teach_eeg_grid_search.py`.

 First, please specify `dataset_load_path`, `model_load_path`, `model_save_path`, `python_package_path`, and `knowledge_path`, accordingly. Note, the folder `teacher_model_folder` has to be placed in your `model_load_path`.
 
Second, execute `project/emotion_analysis_on_mahnob_hci/regression/knowledge_distillation/main_teach_eeg_grid_search.py` using the following two settings, one at a time:

1. Set `case` to `trial` for trial-level shuffling scenario.
2. Set `case` to `loso` for a standard Leave-one-subject-out scenario.