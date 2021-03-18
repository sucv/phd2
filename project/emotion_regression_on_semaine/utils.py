import numpy as np
import os

def generate_trial_info(dataset_info):
    r"""
    Generate unrepeated trial index given the subject index.
        The Semaine has not a existing records on trials of a same subject.
        Therefore, this function is used to generate so that the Subject-trial index is unique for each session.
    :param dataset_info: (dict), the dictionary recording the information of the dataset.
    :return: (dict), a new dictionary having a new key named "trial_id".
    """
    trial_info = np.zeros_like(dataset_info['subject_id'])
    unique_subject_array, count = np.unique(dataset_info['subject_id'], return_counts=True)

    for idx, subject in enumerate(unique_subject_array):
        indices = np.where(dataset_info['subject_id'] == subject)[0]
        trial_info[indices] = np.arange(1, count[idx] + 1, 1)

    dataset_info['trial_id'] = trial_info
    return dataset_info

def read_semaine_xml_for_dataset_info(xml_file, role):
    r"""
    Read the seamin xml log by XPath, so that the session_id, subject_id,
        subject_role and feeltrace boolean can be obtained.
    :param xml_file: (list), the xml file list recording the session information.
    :param role: (list), the role of a subject can be.
    :return: the dictionary obtained, recording the session_id, subject_id,
        subject_role and whether this session has the continuous label.
    """
    contain_continuous_valence_label = 0
    contain_continuous_arousal_label = 0
    feeltrace_bool = 0
    role_dict = {"User": 0, "Operator": 1}

    session_id = xml_file.find('.').attrib['sessionId']

    role_string = './/subject[@role="' + role + '"]'
    subject_id = xml_file.find(role_string).attrib["id"]

    subject_role = role_dict[role]

    target_string = "TU"
    if role == "Operator":
        target_string = "TO"

    feeltrace_string = './/track[@type="AV"]/annotation[@type="FeelTrace"]'

    annotation_tag = xml_file.findall(feeltrace_string)

    if annotation_tag:
        for tag in annotation_tag:
            if "DV.txt" in tag.attrib['filename'] and target_string in tag.attrib['filename']:
                contain_continuous_valence_label = 1
            if "DA.txt" in tag.attrib['filename'] and target_string in tag.attrib['filename']:
                contain_continuous_arousal_label = 1

    if contain_continuous_valence_label == contain_continuous_arousal_label == 1:
        feeltrace_bool = 1

    info = {"session_id": int(session_id),
            "subject_id": int(subject_id),
            "subject_role": int(subject_role),
            "feeltrace_bool": int(feeltrace_bool)}

    return info


def ndarray_to_txt_hauler(input_ndarray, output_txt, column_name, time_interval):
    r"""
    Copy a txt file to a new directory. The old txt file has no headers. Thew new txt file will have
        headers for a column.
    It is designed to re-format the label file in txt format.
    """
    if not os.path.isfile(output_txt):

        with open(output_txt, "w") as txt_file:
            first_line_string = " ".join(column_name) + "\n"
            txt_file.write(first_line_string)

            for index, value in enumerate(input_ndarray[0]):
                string = "{} {}\n".format(time_interval * (index + 1), value)
                txt_file.write(string)