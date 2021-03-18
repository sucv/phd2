import numpy as np
from scipy.stats import pearsonr


class ConcordanceCorrelationCoefficient:
    """
    A class for performing concordance correlation coefficient (CCC) centering. Basically, when multiple continuous labels
    are available, it is not a good choice to perform a direct average. Formally, a Lin's CCC centering has to be done.

    This class is a Pythonic equivalence of CCC centering to the Matlab scripts ("run_gold_standard.m")
        from the AVEC2016 dataset.

    Ref:
        "Lawrence I-Kuei Lin (March 1989).  A concordance correlation coefficient to evaluate reproducibility".
            Biometrics. 45 (1): 255–268. doi:10.2307/2532051. JSTOR 2532051. PMID 2720055.
    """

    def __init__(self, data):
        self.data = data
        if data.shape[0] > data.shape[1]:
            self.data = data.T
        self.rator_number = self.data.shape[0]
        self.combination_list = self.generate_combination_pair()
        self.cnk_matrix = self.generate_cnk_matrix()
        self.ccc = self.calculate_paired_ccc()
        self.agreement = self.calculate_rator_wise_agreement()
        self.mean_data = self.calculate_mean_data()
        self.weight = self.calculate_weight()
        self.centered_data = self.perform_centering()

    def perform_centering(self):
        """
        The centering is done by directly average the shifted and weighted data.
        :return: (ndarray), the centered  data.
        """
        centered_data = self.data - np.repeat(self.mean_data[:, np.newaxis], self.data.shape[1], axis=1) + self.weight
        return centered_data

    def calculate_weight(self):
        """
        The weight of the m continuous labels. It will be used to weight (actually translate) the data when
            performing the final step.
        :return: (float), the weight of the given m continuous labels.
        """
        weight = np.sum((self.mean_data * self.agreement) / np.sum(self.agreement))
        return weight

    def calculate_mean_data(self):
        """
        A directly average of data.
        :return: (ndarray), the averaged data.
        """
        mean_data = np.mean(self.data, axis=1)
        return mean_data

    def generate_combination_pair(self):
        """
        Generate all possible combinations of Cn2.
        :return: (ndarray), the combination list of Cn2.
        """
        n = self.rator_number
        combination_list = []

        for boy in range(n - 1):
            for girl in np.arange(boy + 1, n, 1):
                combination_list.append([boy, girl])

        return np.asarray(combination_list)

    def generate_cnk_matrix(self):
        """
        Generate the Cn2 matrix. The j-th column of the matrix records all the possible candidate
            to the j-th rater. So that for the j-th column, we can acquire all the possible unrepeated
            combination for the j-th rater.
        :return:
        """
        total = self.rator_number
        cnk_matrix = np.zeros((total - 1, total))

        for column in range(total):
            cnk_matrix[:, column] = np.concatenate((np.where(self.combination_list[:, 0] == column)[0],
                                                    np.where(self.combination_list[:, 1] == column)[0]))

        return cnk_matrix.astype(int)

    @staticmethod
    def calculate_ccc(array1, array2):
        """
        Calculate the CCC.
        :param array1: (ndarray), an 1xn array.
        :param array2: (ndarray), another 1xn array.
        :return: the CCC.
        """
        array1_mean = np.mean(array1)
        array2_mean = np.mean(array2)

        array1_var = np.var(array1, ddof=1)
        array2_var = np.var(array2, ddof=1)

        covariance = np.mean((array1 - array1_mean) * (array2 - array2_mean))
        concordance_correlation_coefficient = (2 * covariance) / (
                array1_var + array2_var + (array1_mean - array2_mean) ** 2 + 1e-100)
        return concordance_correlation_coefficient

    def calculate_paired_ccc(self):
        """
        Calculate the CCC for all the pairs from the combination list.
        :return: (ndarray), the CCC for each combinations.
        """
        ccc = np.zeros((self.combination_list.shape[0]))
        for index in range(len(self.combination_list)):
            ccc[index] = self.calculate_ccc(self.data[self.combination_list[index, 0], :],
                                            self.data[self.combination_list[index, 1], :])

        return ccc

    def calculate_rator_wise_agreement(self):
        """
        Calculate the inter-rater CCC agreement.
        :return: (ndarray), a array recording the CCC agreement of each single rater to all the rest raters.
        """

        ccc_agreement = np.zeros(self.rator_number)

        for index in range(self.rator_number):
            ccc_agreement[index] = np.mean(self.ccc[self.cnk_matrix[:, index]])

        return ccc_agreement


class ContinuousMetricsCalculator:
    r"""
    A class to calculate the metrics, usually rmse, pcc, and ccc for continuous regression.
    """

    def __init__(
            self,
            metrics,
            emotional_dimension,
            output_handler,
            continuous_label_handler,
    ):

        # What metrics to calculate.
        self.metrics = metrics

        # What emotional dimensions to consider.
        self.emotional_dimension = emotional_dimension

        # The instances saving the data for evaluation.
        self.output_handler = output_handler
        self.continuous_label_handler = continuous_label_handler

        # Initialize the dictionary for saving the metric results.
        self.metric_record_dict = self.init_metric_record_dict()

    def get_partitionwise_output_and_continuous_label(self):
        return self.output_handler.partitionwise_dict, \
               self.continuous_label_handler.partitionwise_dict

    def get_subjectwise_output_and_continuous_label(self):
        return self.output_handler.subjectwise_dict, \
               self.continuous_label_handler.subjectwise_dict

    def get_sessionwise_output_and_continuous_label(self):
        return self.output_handler.sessionwise_dict, \
               self.continuous_label_handler.sessionwise_dict

    def init_metric_record_dict(self):
        sessionwise_output, _ = self.get_sessionwise_output_and_continuous_label()
        metric_record_dict = {key: [] for key in sessionwise_output}
        return metric_record_dict

    @staticmethod
    def calculator(output, label, metric):
        if metric == "rmse":
            result = np.sqrt(((output - label) ** 2).mean())
        elif metric == "pcc":
            result = pearsonr(output, label)
        elif metric == "ccc":
            result = ConcordanceCorrelationCoefficient.calculate_ccc(output, label)
        else:
            raise ValueError("Metric {} is not defined.".format(metric))
        return result

    def calculate_metrics(self):

        # Load the data for three scenarios.
        # They will all be evaluated.
        sessionwise_output, sessionwise_continuous_label = self.get_sessionwise_output_and_continuous_label()
        subjectwise_output, subjectwise_continuous_label = self.get_subjectwise_output_and_continuous_label()
        partitionwise_output, partitionwise_continuous_label = self.get_partitionwise_output_and_continuous_label()

        for (subject_id, output_list), (_, label_list) in zip(
                sessionwise_output.items(), sessionwise_continuous_label.items()):

            session_record_dict = {key: {} for key in self.emotional_dimension}

            for column, emotion in enumerate(self.emotional_dimension):
                session_number = len(output_list[emotion])

                for metric in self.metrics:
                    session_record_dict[emotion][metric] = {session_id: [] for session_id in range(session_number)}

                for session_id, (output, label) in enumerate(zip(output_list[emotion], label_list[emotion])):
                    # Session-wise evaluation
                    output = np.asarray(output)
                    label = np.asarray(label)

                    for metric in self.metrics:
                        result = self.calculator(output, label, metric)
                        session_record_dict[emotion][metric][session_id].append(result)

                # Subject-wise evaluation
                output = np.asarray(subjectwise_output[subject_id][emotion])
                label = np.asarray(subjectwise_continuous_label[subject_id][emotion])

                for metric in self.metrics:
                    result = self.calculator(output, label, metric)
                    session_record_dict[emotion][metric]['overall'] = []
                    session_record_dict[emotion][metric]['overall'].append(result)

            self.metric_record_dict[subject_id] = session_record_dict

        self.metric_record_dict['overall'] = {}

        # Partition-wise evaluation
        for emotion in self.emotional_dimension:
            partitionwise_dict = {metric: [] for metric in self.metrics}
            output = np.asarray(partitionwise_output[emotion][0])
            label = np.asarray(partitionwise_continuous_label[emotion][0])

            for metric in self.metrics:
                result = self.calculator(output, label, metric)
                partitionwise_dict[metric].append(result)

            self.metric_record_dict['overall'][emotion] = partitionwise_dict
