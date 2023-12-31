#Functions for explaining classifiers that use tabular data (matrices).
import collections
import copy
from functools import partial
import json
import warnings
import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from discretize import QuartileDiscretizer
from discretize import DecileDiscretizer
from discretize import EntropyDiscretizer
from discretize import BaseDiscretizer
import explanation
import explainer_base

class TableDomainMapper(explanation.DomainMapper):
    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None):
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.scaled_row = scaled_row
        self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        out_list = list(zip(self.exp_feature_names,
                            self.feature_values,
                            weights))
        if not show_all:
            out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeTabularExplainer(object):
    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=False,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None):
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous:
            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))
            discretized_training_data = self.discretizer.discretize(
                    training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = explainer_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.scaler = None
        self.class_names = class_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if self.discretizer is not None:
                column = discretized_training_data[:, feature]
            else:
                column = training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(feature_count.items())))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    #Generates explanations for a prediction
    def explain_instance_gmmclust(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         clustered_data = None,
                         regressor='linear', explainer = 'lime'):

        if explainer == 'lime':
            data, inverse = self.__data_inverse(data_row, num_samples)
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

            distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
            ).ravel()

            yss = predict_fn(inverse)
        else:
            data, inverse = self.__data_inverse_gmmclust(data_row, clustered_data)
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_

            distances = sklearn.metrics.pairwise_distances(
                    scaled_data,
                    scaled_data[0].reshape(1, -1),
                    metric=distance_metric
            ).ravel()

            yss = predict_fn(clustered_data)

        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        else:
            try:
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        values = self.convert_and_round(data_row)
           
        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        ret_exp.scaled_data = scaled_data
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection, regressor=regressor)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp
    #Generates a neighborhood around a prediction.
    def __data_inverse(self,
                       data_row,
                       num_samples):
        data = np.zeros((num_samples, data_row.shape[0]))
        categorical_features = range(data_row.shape[0])
        if self.discretizer is None:
            data = self.random_state.normal(
                    0, 1, num_samples * data_row.shape[0]).reshape(
                    num_samples, data_row.shape[0])
            if self.sample_around_instance:
                data = data * self.scaler.scale_ + data_row
            else:
                data = data * self.scaler.scale_ + self.scaler.mean_
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

    def __data_inverse_gmmclust(self,
                       data_row,
                       samples):
        data = np.zeros((samples.shape[0], data_row.shape[0]))
        categorical_features = range(data_row.shape[0])

        first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]

            inverse_column = samples[:,column]
                # self.random_state.choice(values, size=num_samples,
                #                                       replace=True, p=freqs)
            binary_column = np.array([1 if x == data_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        # if self.discretizer is not None:
        #     inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse



        confidence_intervals = []
        Rsquared = []
        intercept = []
        

from sklearn.base import BaseEstimator, RegressorMixin


class Sklearn_Lime(BaseEstimator, RegressorMixin):

    def __init__(self,
                 maxRsquared=1.0,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=False,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 labels=(1,),
                 top_labels=None,
                 num_features=10,
                 num_samples=5000,
                 distance_metric='euclidean',
                 model_regressor=None, clustered_data=None):

        self.maxRsquared =  maxRsquared
        self.mode = mode
        self.training_labels = training_labels
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.kernel_width = kernel_width
        self.kernel = kernel
        self.verbose = verbose
        self.class_names = class_names
        self.feature_selection = feature_selection
        self.discretize_continuous = discretize_continuous
        self.discretizer = discretizer
        self.sample_around_instance = sample_around_instance
        self.random_state = random_state
        self.labels = labels
        self.top_labels = top_labels
        self.num_features = num_features
        self.num_samples = num_samples
        self.distance_metric = distance_metric
        self.model_regressor = model_regressor
        self.clustered_data=clustered_data

    def fit(self, X, y=None):

        self.my_lime = LimeTabularExplainer(training_data=X,
                                               mode=self.mode,training_labels=self.training_labels,
                                               feature_names=self.feature_names,
                                               categorical_features=self.categorical_features,
                                               categorical_names=self.categorical_names,
                                               kernel_width=self.kernel_width,
                                               kernel=self.kernel,
                                               verbose=self.verbose,
                                               class_names=self.class_names,
                                               feature_selection=self.feature_selection,
                                               discretize_continuous=self.discretize_continuous,
                                               discretizer=self.discretizer,
                                               sample_around_instance=self.sample_around_instance,
                                               random_state=self.random_state)
        return self

    def _get_lime_exp(self, data_row, predict_fn):

        self.explanation = self.my_lime.explain_instance_gmmclust(
            data_row,
            predict_fn,
            labels=self.labels,
            top_labels=self.top_labels,
            num_features=self.num_features,
            model_regressor=self.model_regressor,
            clustered_data=self.clustered_data,
            regressor = 'linear',explainer='elime')

        return self.explanation

    def predict(self, X, predict_function, y=None):

        try:
            getattr(self, "my_lime")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        my_exp = self._get_lime_exp(data_row=X, predict_fn=predict_function)

        return my_exp

    def score(self, X, predict_function, y=None, sample_weight=None):
        exp = self.predict(X, predict_function)
    
        max_R_squared = self.maxRsquared

        if exp.score > max_R_squared:
            R_squared = max_R_squared - (exp.score - max_R_squared)
        else:
            R_squared = exp.score

        return R_squared
