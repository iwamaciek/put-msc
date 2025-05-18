import math
import random
import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import os


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score
from Rule import Rule
from Cut import Cut
from CalculateMetrics import calculate_all_metrics, calculate_accuracy

USE_LINE_SEARCH = False
PRE_CHOSEN_K = True
INSTANCE_WEIGHT = 1
R = 5
Rp = 1e-5


class EnderClassifier(BaseEstimator, ClassifierMixin):
    pool = None

    def __init__(self, dataset_name=None, n_rules=100, use_gradient=True, optimized_searching_for_cut=False, nu=1,
                 sampling=1, verbose=True, random_state=42):
        self.dataset_name = dataset_name
        self.n_rules = n_rules
        self.rules = []

        self.use_gradient = use_gradient
        self.nu = nu
        self.sampling = sampling

        self.verbose = verbose
        self.random_state = random_state
        random.seed(random_state)

        self.optimized_searching_for_cut = optimized_searching_for_cut
        self.history = {'accuracy': [],
                        'mean_absolute_error': [],
                        'accuracy_test': [],
                        'mean_absolute_error_test': []}

        self.is_fitted_ = False

        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.attribute_names = None
        self.num_classes = None
        self.value_of_f = None
        self.probability = None
        self.default_rule = None
        self.covered_instances = None
        self.last_index_computation_of_empirical_risk = None
        self.gradient = None
        self.hessian = None
        self.gradients = None
        self.hessians = None
        self.inverted_list = None
        self.indices_for_better_cuts = None
        self.max_k = None
        self.effective_rules = None

        plt.style.use('ggplot')

    def fit(self, X, y, X_test=None, y_test=None):
        self.attribute_names = X.columns
        X, y = check_X_y(X, y)
        if X_test is not None and y_test is not None:
            X_test, y_test = check_X_y(X_test, y_test)
            self.X_test = X_test
            self.y_test = y_test
        self.X = X
        self.y = y

        self.num_classes = len(set(y))
        self.value_of_f = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]

        self.create_rules(X)

        self.is_fitted_ = True

        return None

    def create_rules(self, X):
        self.create_inverted_list(X)
        self.covered_instances = [1 for _ in range(len(X))]

        self.default_rule = self.create_default_rule()
        self.rules = []
        self.update_value_of_f(self.default_rule)
        if self.verbose: print("Default rule:", self.default_rule)
        i_rule = 0
        while i_rule < self.n_rules:
            if self.verbose:
                print('####################################################################################')
                print(f"Rule: {i_rule + 1}")
            self.covered_instances = self.resampling()
            rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                self.rules.append(rule)
                i_rule += 1

    def resampling(self):
        count = Counter(self.y)
        total = len(self.y)
        no_examples_to_use = math.ceil(len(self.y) * self.sampling)

        ones_allocation = {key: round((value / total) * no_examples_to_use) for key, value in count.items()}

        allocated_ones = sum(ones_allocation.values())
        difference = no_examples_to_use - allocated_ones
        keys = list(ones_allocation.keys())

        while difference != 0:
            for key in keys:
                if difference == 0:
                    break
                if difference > 0:
                    ones_allocation[key] += 1
                    difference -= 1
                else:
                    if ones_allocation[key] > 0:
                        ones_allocation[key] -= 1
                        difference += 1

        result = [0] * len(self.y)

        for key, num_ones in ones_allocation.items():
            indices = [i for i, x in enumerate(self.y) if x == key]
            selected_indices = random.sample(indices, num_ones)
            for index in selected_indices:
                result[index] = 1
        return result

    def create_rule(self):
        self.initialize_for_rule()
        rule = Rule()

        best_cut = Cut()
        best_cut.empirical_risk = 0

        creating = True
        EPSILON = 1e-8
        count = 0
        while creating:
            count += 1
            best_attribute = -1
            cut = Cut()
            for attribute in range(len(self.X[0])):
                cut = self.find_best_cut(attribute)
                if cut.empirical_risk < best_cut.empirical_risk - EPSILON:
                    best_cut = cut
                    best_attribute = attribute

            if best_attribute == -1 or not best_cut.exists:
                creating = False
            else:
                rule.add_condition(best_attribute, best_cut.value, best_cut.direction,
                                   self.attribute_names[best_attribute])
                self.mark_covered_instances(best_attribute, best_cut)
        if best_cut.exists:

            decision = self.compute_decision()
            if decision is None:
                return None

            decision = [dec * self.nu for dec in decision]

            rule.decision = decision
            if self.verbose:
                for i_condition in range(len(rule.conditions)):
                    if rule.conditions[i_condition][1] == -99999999999999999:
                        print(f'\t{rule.attribute_names[i_condition]} <= {rule.conditions[i_condition][2]}')
                    elif rule.conditions[i_condition][2] == 99999999999999999:
                        print(f'\t{rule.attribute_names[i_condition]} >= {rule.conditions[i_condition][1]}')
                    else:
                        print(
                            f'\t{rule.attribute_names[i_condition]} in [{rule.conditions[i_condition][1]}, {rule.conditions[i_condition][2]}]')
                max_weight = max(rule.decision)
                print(f'=> vote for class {rule.decision.index(max_weight)} with weight {max_weight}')
                print(rule.decision)
                print()
            return rule
        else:
            return None

    def find_best_cut(self, attribute):
        best_cut = Cut()
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0

        temp_empirical_risk = 0

        GREATER_EQUAL = 1
        LESS_EQUAL = -1
        EPSILON = 1e-8

        empirical_risks = []
        indices_to_check = []
        for cut_direction in [-1, 1]:
            self.initialize_for_cut()

            if self.optimized_searching_for_cut == 2:
                if len(empirical_risks) == 0:
                    i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
                    previous_position = self.inverted_list[attribute][i]
                    previous_value = self.X[previous_position][attribute]
                    previous_class = self.y[previous_position]
                    count = i - 1
                    while (cut_direction == GREATER_EQUAL and i >= 0) or (
                            cut_direction != GREATER_EQUAL and i < len(self.X)):
                        curr_position = self.inverted_list[attribute][i]
                        if self.covered_instances[curr_position] == 1:
                            count += 1
                            curr_value = self.X[curr_position][attribute]
                            weight = 1

                            if previous_value != curr_value:
                                if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                    best_cut.direction = cut_direction
                                    best_cut.value = (previous_value + curr_value) / 2
                                    best_cut.empirical_risk = temp_empirical_risk
                                    best_cut.exists = True

                            temp_empirical_risk, added_risk = self.compute_current_empirical_risk_optimized(
                                curr_position, self.covered_instances[curr_position] * weight)

                            empirical_risks.append([added_risk, previous_value, curr_value, count])
                            current_class = self.y[curr_position]
                            if previous_class != current_class: indices_to_check.append(count)

                            previous_class = current_class
                            previous_value = self.X[curr_position][attribute]

                        i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
                else:
                    current_risk = 0
                    risks = []
                    for (added_risk, previous_value, curr_value, i) in empirical_risks[::-1]:
                        current_risk += added_risk
                        risks.append(current_risk)
                    risks = risks[::-1]
                    for j in indices_to_check:
                        added_risk, previous_value, curr_value, i = empirical_risks[j]
                        if previous_value != curr_value:
                            if risks[j] < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = 1
                                best_cut.value = (previous_value + curr_value) / 2
                                best_cut.empirical_risk = risks[j]
                                best_cut.exists = True

            elif self.optimized_searching_for_cut == 1:
                if len(empirical_risks) == 0:
                    i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
                    previous_position = self.inverted_list[attribute][i]
                    previous_value = self.X[previous_position][attribute]

                    while (cut_direction == GREATER_EQUAL and i >= 0) or (
                            cut_direction != GREATER_EQUAL and i < len(self.X)):
                        curr_position = self.inverted_list[attribute][i]
                        if self.covered_instances[curr_position] == 1:
                            curr_value = self.X[curr_position][attribute]
                            weight = 1

                            if previous_value != curr_value:
                                if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                    best_cut.direction = cut_direction
                                    best_cut.value = (previous_value + curr_value) / 2
                                    best_cut.empirical_risk = temp_empirical_risk
                                    best_cut.exists = True

                            temp_empirical_risk, added_risk = self.compute_current_empirical_risk_optimized(
                                curr_position, self.covered_instances[curr_position] * weight)
                            empirical_risks.append([added_risk, previous_value, curr_value])

                            previous_value = self.X[curr_position][attribute]
                        i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
                else:
                    risk = 0
                    for j, (added_risk, previous_value, curr_value) in enumerate(empirical_risks[::-1]):
                        risk += added_risk
                        if previous_value != curr_value:
                            if risk < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = 1
                                best_cut.value = (previous_value + curr_value) / 2
                                best_cut.empirical_risk = risk
                                best_cut.exists = True

            elif self.optimized_searching_for_cut == 0:
                i = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0
                previous_position = self.inverted_list[attribute][i]
                previous_value = self.X[previous_position][attribute]
                count = 0
                while (cut_direction == GREATER_EQUAL and i >= 0) or (
                        cut_direction != GREATER_EQUAL and i < len(self.X)):
                    count += 1
                    curr_position = self.inverted_list[attribute][i]
                    if self.covered_instances[curr_position] == 1:
                        if True:
                            curr_value = self.X[curr_position][attribute]
                            weight = 1

                            if previous_value != curr_value:
                                if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                    best_cut.direction = cut_direction
                                    best_cut.value = (previous_value + curr_value) / 2
                                    best_cut.empirical_risk = temp_empirical_risk
                                    best_cut.exists = True

                            temp_empirical_risk = self.compute_current_empirical_risk(
                                curr_position, self.covered_instances[curr_position] * weight)

                            previous_value = self.X[curr_position][attribute]
                    i = i - 1 if cut_direction == GREATER_EQUAL else i + 1

        return best_cut

    def mark_covered_instances(self, best_attribute, cut):
        for i in range(len(self.X)):
            if self.covered_instances[i] != -1:
                value = self.X[i][best_attribute]
                if (value < cut.value and cut.direction == 1) or (value > cut.value and cut.direction == -1):
                    self.covered_instances[i] = -1

    def initialize_for_cut(self):
        self.gradient = 0
        self.hessian = R
        self.gradients = [0 for _ in range(self.num_classes)]
        self.hessians = [R for _ in range(self.num_classes)]

    def compute_current_empirical_risk_optimized(self, next_position, weight):
        if PRE_CHOSEN_K:
            gradient_difference = 0
            if self.y[next_position] == self.max_k:
                self.gradient += INSTANCE_WEIGHT * weight
                gradient_difference += INSTANCE_WEIGHT * weight
            self.gradient -= INSTANCE_WEIGHT * weight * self.probability[next_position][self.max_k]
            gradient_difference -= INSTANCE_WEIGHT * weight * self.probability[next_position][self.max_k]
            if self.use_gradient:
                return -self.gradient, -gradient_difference
            else:
                raphson_now = - (self.gradient - gradient_difference) * abs(self.gradient - gradient_difference) / self.hessian
                self.hessian += INSTANCE_WEIGHT * weight * (Rp + self.probability[next_position][self.max_k] * (
                        1 - self.probability[next_position][self.max_k]))
                return - self.gradient * abs(self.gradient) / self.hessian, - self.gradient * abs(self.gradient) / self.hessian - raphson_now
        else:
            raise

    def compute_current_empirical_risk(self, next_position, weight):
        if PRE_CHOSEN_K:
            if self.y[next_position] == self.max_k:
                self.gradient += INSTANCE_WEIGHT * weight
            self.gradient -= INSTANCE_WEIGHT * weight * self.probability[next_position][self.max_k]
            if self.use_gradient:
                return -self.gradient
            else:
                self.hessian += INSTANCE_WEIGHT * weight * (Rp + self.probability[next_position][self.max_k] * (
                        1 - self.probability[next_position][self.max_k]))
                return - self.gradient * abs(self.gradient) / self.hessian
        else:
            raise

    def create_default_rule(self):
        self.initialize_for_rule()
        decision = self.compute_decision()
        for i in range(self.num_classes):
            decision[i] *= self.nu
        return decision

    def compute_decision(self):
        if PRE_CHOSEN_K:
            hessian = R
            gradient = 0

            for i in range(len(self.covered_instances)):
                if self.covered_instances[i] >= 0:
                    if self.y[i] == self.max_k:
                        gradient += INSTANCE_WEIGHT
                    gradient -= INSTANCE_WEIGHT * self.probability[i][self.max_k]
                    hessian += INSTANCE_WEIGHT * (
                            Rp + self.probability[i][self.max_k] * (1 - self.probability[i][self.max_k]))

            if gradient < 0:
                return None

            alpha_nr = gradient / hessian
            decision = [- alpha_nr / self.num_classes for _ in range(self.num_classes)]
            decision[self.max_k] = alpha_nr * (self.num_classes - 1) / self.num_classes
            return decision
        else:
            raise

    def initialize_for_rule(self):
        if PRE_CHOSEN_K:
            self.gradients = [0 for _ in range(self.num_classes)]
            self.hessians = [R for _ in range(self.num_classes)]
        else:
            raise

        for i in range(len(self.X)):
            if self.covered_instances[i] >= 1:
                norm = 0
                for k in range(self.num_classes):
                    self.probability[i][k] = math.exp(self.value_of_f[i][k])
                    norm += self.probability[i][k]
                for k in range(self.num_classes):
                    self.probability[i][k] /= norm
                    if PRE_CHOSEN_K:
                        self.gradients[k] -= INSTANCE_WEIGHT * self.probability[i][k]
                        self.hessians[k] += INSTANCE_WEIGHT * (
                                Rp + self.probability[i][k] * (1 - self.probability[i][k]))
                if PRE_CHOSEN_K:
                    self.gradients[self.y[i]] += INSTANCE_WEIGHT

        if PRE_CHOSEN_K:
            self.max_k = 0
            if self.use_gradient:
                for k in range(1, self.num_classes):
                    if self.gradients[k] > self.gradients[self.max_k]:
                        self.max_k = k
            else:
                for k in range(1, self.num_classes):
                    if self.gradients[k] / self.hessians[k] ** .5 > self.gradients[self.max_k] / self.hessians[self.max_k] ** .5:
                        self.max_k = k

    def create_inverted_list(self, X):
        import numpy as np
        X = np.array(X)
        sorted_indices = np.argsort(X, axis=0)
        self.inverted_list = sorted_indices.T
        temp = self.inverted_list.copy()
        temp = np.array([[self.y[temp[i][j]] for j in range(len(temp[0]))] for i in range(len(temp))])

    def update_value_of_f(self, decision):
        for i in range(len(self.X)):
            if self.covered_instances[i] >= 0:
                for k in range(self.num_classes):
                    self.value_of_f[i][k] += decision[k]

    def predict(self, X, use_effective_rules=True):
        X = check_array(X)
        predictions = [self.predict_instance(x, use_effective_rules) for x in X]
        return predictions

    def predict_instance(self, x, use_effective_rules):
        value_of_f_instance = np.array(self.default_rule)
        rules = self.rules
        for rule in rules:
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance

    def predict_with_specific_rules(self, X, rule_indices):
        X = check_array(X)
        preds = []
        for x in X:
            pred = np.array(self.default_rule)
            for rule_index in rule_indices:
                pred += np.array(self.rules[rule_index].classify_instance(x))
            preds.append(pred)
        return np.array(preds)

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')

        X, y = check_X_y(X, y)

        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return accuracy

    def evaluate_all_rules(self):
        from tqdm import tqdm

        predictions_train = np.array([self.default_rule for _ in range(len(self.X))])
        metrics = calculate_all_metrics(self.y, predictions_train)
        self.history['accuracy'] = [metrics['accuracy']]
        self.history['f1'] = [metrics['f1']]
        self.history['mean_absolute_error'] = [metrics['mean_absolute_error']]
        for i_rule in tqdm(range(self.n_rules)):
            for i_x, x in enumerate(self.X):
                predictions_train[i_x] += np.array(self.rules[i_rule].classify_instance(x))
            metrics = calculate_all_metrics(self.y, predictions_train)
            self.history['accuracy'].append(metrics['accuracy'])
            self.history['f1'].append(metrics['f1'])
            self.history['mean_absolute_error'].append(metrics['mean_absolute_error'])

        if self.X_test is not None and self.y_test is not None:
            predictions_test = np.array([self.default_rule for _ in range(len(self.X_test))])
            metrics = calculate_all_metrics(self.y_test, predictions_test)
            self.history['accuracy_test'] = [metrics['accuracy']]
            self.history['f1_test'] = [metrics['f1']]
            self.history['mean_absolute_error_test'] = [metrics['mean_absolute_error']]
            for i_rule in tqdm(range(self.n_rules)):
                for i_x, x in enumerate(self.X_test):
                    predictions_test[i_x] += np.array(self.rules[i_rule].classify_instance(x))
                metrics = calculate_all_metrics(self.y_test, predictions_test)
                self.history['accuracy_test'].append(metrics['accuracy'])
                self.history['f1_test'].append(metrics['f1'])
                self.history['mean_absolute_error_test'].append(metrics['mean_absolute_error'])

    def prune_rules(self, regressor, **kwargs):
        rule_feature_matrix_train = [[0 if rule.classify_instance(x)[0] == 0 else 1 for rule in self.rules] for x in
                                     kwargs['x_tr'].to_numpy()]
        rule_feature_matrix_test = [[0 if rule.classify_instance(x)[0] == 0 else 1 for rule in self.rules] for x in
                                    kwargs['x_te'].to_numpy()]

        if regressor == 'MyIdeaWrapper':
            self.my_idea_wrapper_pruning(**kwargs)
            return
        elif regressor == 'Wrapper':
            self.wrapper_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        elif regressor == 'Filter':
            self.filter_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return
        elif regressor == "Embedded":
            self.embedded_pruning(rule_feature_matrix_train, rule_feature_matrix_test, **kwargs)
            return

    def filter_pruning_one_method(self, method, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.feature_selection import SelectKBest
        from sklearn.linear_model import LogisticRegression

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for i in tqdm(range(1, len(self.rules) + 1)):
            selector = SelectKBest(method, k=i)

            selector.fit(rule_feature_matrix_train, y_train)
            selected_rules_bool = selector.get_support()
            chosen_rules = []
            for rule_index, rule_is_chosen in enumerate(selected_rules_bool):
                if rule_is_chosen:
                    chosen_rules.append(rule_index)

            classifier = LogisticRegression(
                # multi_class='auto', penalty='l1', solver='saga',
                random_state=self.random_state,
                max_iter=200
            )

            X_train_new = np.array(rule_feature_matrix_train)[:, chosen_rules]
            X_test_new = np.array(rule_feature_matrix_test)[:, chosen_rules]
            classifier.fit(X_train_new, y_train)
            y_train_preds = classifier.predict(X_train_new)
            y_test_preds = classifier.predict(X_test_new)
            current_train_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)
            current_test_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)
            train_acc.append(current_train_acc)
            test_acc.append(current_test_acc)

        return train_acc, test_acc

    def filter_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, verbose=True, **kwargs):
        from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

        print("\tChi 2:")
        chi2_train_acc, chi2_test_acc = self.filter_pruning_one_method(chi2, rule_feature_matrix_train,
                                                                       rule_feature_matrix_test, **kwargs)

        print("\tAnova:")
        anova_train_acc, anova_test_acc = self.filter_pruning_one_method(f_classif, rule_feature_matrix_train,
                                                                         rule_feature_matrix_test, **kwargs)
        print("\tMutual Info:")
        mutual_info_train_acc, mutual_info_test_acc = self.filter_pruning_one_method(mutual_info_classif,
                                                                                     rule_feature_matrix_train,
                                                                                     rule_feature_matrix_test, **kwargs)

        if verbose:
            plt.figure(figsize=(14, 10))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), chi2_train_acc, label='Pruned rules Filter Chi-squared, train dataset',
                     c='r')
            plt.plot(list(range(self.n_rules + 1)), anova_train_acc, label='Pruned rules Filter ANOVA, train dataset',
                     c='y')
            plt.plot(list(range(self.n_rules + 1)), mutual_info_train_acc,
                     label='Pruned rules Filter Mutual info, train dataset', c='k')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), chi2_test_acc, label='Pruned rules Filter Chi-squared, test dataset',
                     c='r', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), anova_test_acc, label='Pruned rules Filter ANOVA, test dataset',
                     c='y', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), mutual_info_test_acc,
                     label='Pruned rules Filter Mutual info, test dataset', c='k', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Filter Methods Against Baseline Rules', wrap=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'Filter',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
            )
            plt.show()
        return

    def my_idea_wrapper_pruning(self, verbose=True, **kwargs):

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        # UPWARD
        print("\tUpward")
        rules_indices_upward = []
        train_acc_upward = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc_upward = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for _ in tqdm(range(self.n_rules)):
            max_acc = -1
            max_acc_index = -1
            for i in range(self.n_rules):
                if i in rules_indices_upward:
                    continue
                y_preds = self.predict_with_specific_rules(X_train, rules_indices_upward + [i])
                current_acc = calculate_accuracy(y_train, y_preds)
                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = i
            rules_indices_upward.append(max_acc_index)
            train_acc_upward.append(max_acc)
            test_acc_upward.append(
                calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, rules_indices_upward)))
        # DOWNWARD
        print("\tDownward")
        rules_indices_downward = []
        train_acc_downward = [calculate_accuracy(y_train, self.predict(X_train, use_effective_rules=False))]
        test_acc_downward = [calculate_accuracy(y_test, self.predict(X_test, use_effective_rules=False))]
        indices_in_use = list(range(self.n_rules))
        for _ in tqdm(range(self.n_rules)):
            max_acc = 0
            max_acc_index = -1
            for rule_index in indices_in_use:
                temp_indices_in_use = indices_in_use.copy()
                temp_indices_in_use.remove(rule_index)
                y_preds = self.predict_with_specific_rules(X_train, temp_indices_in_use)
                current_acc = calculate_accuracy(y_train, y_preds)
                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = rule_index
            indices_in_use.remove(max_acc_index)
            rules_indices_downward.insert(0, max_acc_index)
            train_acc_downward.insert(0, max_acc)
            test_acc_downward.insert(0, calculate_accuracy(y_test,
                                                           self.predict_with_specific_rules(X_test, indices_in_use)))

        # train_acc_upward = [0.50125, 0.7, 0.76375, 0.7825, 0.77875, 0.78875, 0.78875, 0.78875, 0.80375, 0.795, 0.79625, 0.8025, 0.80125, 0.8025, 0.81125, 0.81625, 0.81625, 0.81125, 0.81, 0.81125, 0.81875, 0.8125, 0.8075, 0.8225, 0.83125, 0.82875, 0.835, 0.835, 0.84375, 0.8375, 0.8375, 0.84, 0.835, 0.83625, 0.83625, 0.83625, 0.83875, 0.835, 0.835, 0.8325, 0.825, 0.83875, 0.83625, 0.83875, 0.8425, 0.845, 0.84, 0.8425, 0.8425, 0.84875, 0.84625, 0.83875, 0.84875, 0.85, 0.85125, 0.85375, 0.8475, 0.845, 0.84125, 0.8425, 0.8425, 0.84375, 0.83875, 0.84, 0.8375, 0.8425, 0.84375, 0.8425, 0.84375, 0.84125, 0.8425, 0.8375, 0.83875, 0.835, 0.85, 0.85, 0.8575, 0.85375, 0.85625, 0.865, 0.86375, 0.865, 0.8625, 0.865, 0.86625, 0.85875, 0.8625, 0.86, 0.8675, 0.86125, 0.85875, 0.8575, 0.8625, 0.86375, 0.85875, 0.85625, 0.855, 0.85125, 0.84875, 0.855, 0.8575]
        if verbose:
            # print(f"Rules order up: {rules_indices_upward} Accuracies: {test_acc_upward}")
            # print(f"Rules order down: {rules_indices_downward} Accuracies: {test_acc_downward}")
            plt.figure(figsize=(20, 14))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), train_acc_upward,
                     label='Proposed forward selection, train dataset', c='g')
            plt.plot(list(range(self.n_rules + 1)), train_acc_downward,
                     label='Proposed backward elimination, train dataset', c='y')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_upward,
                     label='Proposed forward selection, test dataset', c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_downward,
                     label='Proposed backward elimination, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Proposed Methods Against Baseline Rules')
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'MyIdeaWrapper',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
            )
            plt.show()
        return

    def wrapper_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, verbose=True, **kwargs):
        from sklearn.linear_model import LogisticRegression

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']
        # UPWARD
        print("\tUpward")
        rules_indices_upward = []
        train_acc_upward = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc_upward = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        for _ in tqdm(range(self.n_rules)):
            max_acc = -1
            max_acc_index = -1
            max_classifier = None
            for i in range(self.n_rules):
                if i in rules_indices_upward:
                    continue
                X_train_new = np.array(rule_feature_matrix_train)[:, rules_indices_upward + [i]]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga',
                                                random_state=self.random_state)

                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)

                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = i
                    max_classifier = classifier
            rules_indices_upward.append(max_acc_index)
            X_test_new = np.array(rule_feature_matrix_test)[:, rules_indices_upward]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc_upward.append(max_acc)
            test_acc_upward.append(max_acc_test)

        # # DOWNWARD
        print("\tDownward")
        rules_indices_downward = []
        classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga', random_state=self.random_state)
        classifier.fit(np.array(rule_feature_matrix_train), y_train)
        y_train_preds = classifier.predict(np.array(rule_feature_matrix_train))
        y_test_preds = classifier.predict(np.array(rule_feature_matrix_test))
        train_acc_downward = [sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)]
        test_acc_downward = [sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)]
        indices_in_use = list(range(self.n_rules))
        for _ in tqdm(range(self.n_rules - 1)):
            max_acc = 0
            max_acc_index = -1
            max_classifier = None
            for rule_index in indices_in_use:
                temp_indices_in_use = indices_in_use.copy()
                temp_indices_in_use.remove(rule_index)
                X_train_new = np.array(rule_feature_matrix_train)[:, temp_indices_in_use]

                classifier = LogisticRegression(multi_class='auto', penalty='l1', solver='saga',
                                                random_state=self.random_state)
                classifier.fit(X_train_new, y_train)
                y_train_preds = classifier.predict(X_train_new)
                current_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)

                if current_acc > max_acc:
                    max_acc = current_acc
                    max_acc_index = rule_index
                    max_classifier = classifier

            indices_in_use.remove(max_acc_index)
            rules_indices_downward.insert(0, max_acc_index)
            X_test_new = np.array(rule_feature_matrix_test)[:, indices_in_use]
            y_test_preds = max_classifier.predict(X_test_new)
            max_acc_test = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc_downward.insert(0, max_acc)
            test_acc_downward.insert(0, max_acc_test)
        train_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))
        test_acc_downward.insert(0, calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, [])))
        #
        # rules_indices_upward = [45, 55, 6, 99, 52, 4, 81, 64, 72, 40, 53, 73, 67, 13, 95, 89, 62, 85, 15, 7, 68, 19, 5, 49, 61, 63, 71, 66, 46, 2, 43, 33, 44, 22, 92, 0, 70, 11, 9, 20, 75, 34, 48, 58, 59, 88, 65, 41, 25, 24, 35, 82, 77, 27, 18, 78, 74, 98, 29, 47, 38, 54, 17, 51, 14, 50, 16, 86, 96, 87, 21, 39, 80, 28, 1, 32, 12, 91, 36, 30, 10, 57, 84, 8, 93, 31, 94, 79, 83, 56, 23, 3, 42, 97, 76, 60, 69, 26, 90, 37]
        # test_acc_upward = [0.50125, 0.695, 0.7125, 0.7625, 0.78375, 0.8125, 0.81125, 0.815, 0.8125, 0.81625, 0.8125, 0.7975, 0.7975, 0.82375, 0.8225, 0.82125, 0.82625, 0.82375, 0.82375, 0.82375, 0.82625, 0.81875, 0.81875, 0.82, 0.81625, 0.8225, 0.8375, 0.8325, 0.83375, 0.83875, 0.84375, 0.84125, 0.84125, 0.84375, 0.83625, 0.84375, 0.8375, 0.8375, 0.84, 0.83875, 0.84125, 0.84, 0.84, 0.84, 0.84, 0.84125, 0.84375, 0.845, 0.845, 0.8475, 0.845, 0.8475, 0.85, 0.85375, 0.85375, 0.8525, 0.85125, 0.84625, 0.845, 0.84875, 0.84875, 0.84875, 0.85125, 0.85, 0.85, 0.8425, 0.83625, 0.83625, 0.83625, 0.83625, 0.84125, 0.84, 0.83875, 0.8375, 0.84375, 0.845, 0.845, 0.84125, 0.8375, 0.8375, 0.84125, 0.8425, 0.84, 0.83125, 0.83, 0.83125, 0.835, 0.83, 0.83, 0.82625, 0.82375, 0.83, 0.82875, 0.8275, 0.83125, 0.83, 0.83125, 0.83, 0.83875, 0.84375, 0.83875]
        # train_acc_upward = [0.5009375, 0.7115625, 0.74125, 0.7803125, 0.7971875, 0.81, 0.8278125, 0.83375, 0.8396875, 0.8425, 0.8525, 0.86, 0.8659375, 0.8671875, 0.870625, 0.8740625, 0.87625, 0.8775, 0.8778125, 0.8778125, 0.8775, 0.8775, 0.88, 0.880625, 0.8809375, 0.88125, 0.883125, 0.8834375, 0.885, 0.8875, 0.890625, 0.8928125, 0.8953125, 0.8975, 0.9, 0.9025, 0.904375, 0.9071875, 0.9103125, 0.9103125, 0.910625, 0.9115625, 0.911875, 0.911875, 0.911875, 0.9115625, 0.913125, 0.9134375, 0.9128125, 0.9134375, 0.9121875, 0.9128125, 0.915625, 0.9159375, 0.91625, 0.9159375, 0.915625, 0.9159375, 0.919375, 0.92, 0.920625, 0.92, 0.92, 0.9203125, 0.92, 0.9190625, 0.9209375, 0.920625, 0.92, 0.9225, 0.9253125, 0.9253125, 0.925625, 0.9246875, 0.9253125, 0.9259375, 0.92625, 0.92625, 0.9259375, 0.92625, 0.925625, 0.925625, 0.9259375, 0.9275, 0.9271875, 0.92625, 0.92625, 0.9259375, 0.9290625, 0.92875, 0.9303125, 0.9303125, 0.93125, 0.930625, 0.93125, 0.9315625, 0.9315625, 0.9309375, 0.93, 0.9303125, 0.92875]
        #
        #
        # rules_indices_downward = [55, 84, 89, 97, 46, 80, 71, 51, 64, 96, 26, 90, 48, 25, 81, 74, 43, 49, 83, 22, 59, 54, 60, 98, 53, 70, 6, 63, 78, 82, 85, 94, 87, 10, 61, 42, 0, 50, 66, 77, 40, 75, 92, 88, 79, 36, 86, 69, 41, 24, 73, 23, 72, 12, 1, 11, 99, 7, 95, 20, 91, 27, 38, 5, 28, 58, 30, 56, 15, 44, 65, 57, 18, 39, 47, 33, 52, 35, 31, 93, 29, 14, 62, 17, 4, 13, 34, 21, 16, 9, 8, 68, 32, 76, 3, 2, 19, 37, 67]
        # test_acc_downward = [0.5009375, 0.695, 0.7125, 0.72625, 0.74875, 0.74875, 0.7675, 0.76875, 0.75875, 0.765, 0.79, 0.78125, 0.78, 0.775, 0.77125, 0.79625, 0.805, 0.8125, 0.80375, 0.81375, 0.8125, 0.81875, 0.81875, 0.815, 0.82375, 0.825, 0.81875, 0.82125, 0.825, 0.83, 0.83125, 0.82875, 0.82625, 0.82125, 0.82, 0.81875, 0.82, 0.82125, 0.82375, 0.82625, 0.82625, 0.82875, 0.82375, 0.8275, 0.83625, 0.83375, 0.835, 0.8325, 0.8325, 0.8325, 0.83375, 0.8375, 0.83375, 0.83625, 0.83375, 0.83625, 0.835, 0.8375, 0.8375, 0.83125, 0.82875, 0.83, 0.83, 0.83375, 0.8375, 0.8375, 0.8375, 0.8375, 0.83625, 0.83375, 0.83625, 0.83375, 0.835, 0.83625, 0.83625, 0.83125, 0.83125, 0.83, 0.83375, 0.835, 0.835, 0.835, 0.835, 0.835, 0.83625, 0.83625, 0.835, 0.835, 0.83, 0.83, 0.83, 0.83, 0.83125, 0.83, 0.8325, 0.8325, 0.83375, 0.83375, 0.83375, 0.835, 0.83875]
        # train_acc_downward = [0.5009375, 0.7115625, 0.74125, 0.7659375, 0.7878125, 0.7878125, 0.805625, 0.808125, 0.8178125, 0.8296875, 0.8359375, 0.8428125, 0.8496875, 0.8540625, 0.859375, 0.8678125, 0.8709375, 0.875, 0.87875, 0.881875, 0.8865625, 0.890625, 0.8934375, 0.89875, 0.8996875, 0.9021875, 0.9053125, 0.90875, 0.91125, 0.91375, 0.9140625, 0.9159375, 0.9178125, 0.9184375, 0.919375, 0.920625, 0.9203125, 0.9215625, 0.923125, 0.924375, 0.9265625, 0.9275, 0.9290625, 0.929375, 0.9309375, 0.93125, 0.930625, 0.933125, 0.931875, 0.9328125, 0.9346875, 0.9340625, 0.9340625, 0.933125, 0.9334375, 0.9328125, 0.93375, 0.9340625, 0.9334375, 0.935, 0.935, 0.935625, 0.9353125, 0.9340625, 0.9340625, 0.9334375, 0.934375, 0.934375, 0.935, 0.9353125, 0.9353125, 0.935625, 0.9359375, 0.935625, 0.935625, 0.935625, 0.9359375, 0.9359375, 0.935625, 0.9359375, 0.93625, 0.93625, 0.9365625, 0.9365625, 0.93625, 0.93625, 0.93625, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.9359375, 0.935625, 0.9346875, 0.9340625, 0.9340625, 0.9340625, 0.9328125, 0.93125, 0.92875]

        if verbose:
            print("history train", self.history['accuracy'])
            print("history test", self.history['accuracy_test'])
            print(f"Rules order: {rules_indices_upward} Accuracies test: {test_acc_upward}, Accuracies train: {train_acc_upward}")
            print(f"Rules order: {rules_indices_downward} Accuracies test: {test_acc_downward}, Accuracies train {train_acc_downward}")
            plt.figure(figsize=(14, 10))
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy'], label='Baseline rules, train dataset', c='b')
            plt.plot(list(range(self.n_rules + 1)), train_acc_upward,
                     label='Forward selection, train dataset', c='g')
            plt.plot(list(range(self.n_rules + 1)), train_acc_downward,
                     label='Backward elimination, train dataset', c='y')
            plt.plot(list(range(self.n_rules + 1)), self.history['accuracy_test'], label='Baseline rules, test dataset',
                     c='b', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_upward, label='Forward selection, test dataset',
                     c='g', linestyle='dashed')
            plt.plot(list(range(self.n_rules + 1)), test_acc_downward,
                     label='Backward elimination, test dataset', c='y', linestyle='dashed')
            plt.legend()
            plt.xlabel("Rules")
            plt.ylabel("Accuracy")
            plt.title('Comparison Between Wrapper Methods Against Baseline Rules', wrap=True)
            plt.tight_layout()
            plt.savefig(
                os.path.join('Plots',
                             'Pruning',
                             'Wrapper',
                             f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')

            )
            plt.show()
        return

    def embedded_pruning(self, rule_feature_matrix_train, rule_feature_matrix_test, **kwargs):
        from sklearn.linear_model import LogisticRegression

        X_train = kwargs['x_tr']
        X_test = kwargs['x_te']
        y_train = kwargs['y_tr']
        y_test = kwargs['y_te']

        from matplotlib import pyplot as plt
        import os

        alphas = [1.5 ** x for x in range(-20, 10)]
        train_acc = [calculate_accuracy(y_train, self.predict_with_specific_rules(X_train, []))]
        test_acc = [calculate_accuracy(y_test, self.predict_with_specific_rules(X_test, []))]
        active_rule_number = [0]
        for alpha in tqdm(alphas):
            pruning_model = LogisticRegression(multi_class='multinomial', penalty='l1', solver='saga', C=alpha,
                                               max_iter=10000)
            pruning_model.fit(rule_feature_matrix_train, y_train)

            y_train_preds = pruning_model.predict(rule_feature_matrix_train)
            y_test_preds = pruning_model.predict(rule_feature_matrix_test)

            current_train_acc = sum([y == y_p for y, y_p in zip(y_train, y_train_preds)]) / len(y_train)
            current_test_acc = sum([y == y_p for y, y_p in zip(y_test, y_test_preds)]) / len(y_test)

            train_acc.append(current_train_acc)
            test_acc.append(current_test_acc)

            coefs = pruning_model.coef_
            coefs = np.abs(coefs)
            coefs[(coefs > -0.01) & (coefs < 0.01)] = 0
            active_rule_number.append(np.count_nonzero(np.sum(coefs, axis=0)))

        alphas = [0] + alphas
        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        ax[0].set_title("Impact of Regularization\non Ensemble Size", wrap=True)
        ax[0].plot(alphas, active_rule_number)
        ax[0].scatter(alphas, active_rule_number)
        ax[0].set_xscale('log')
        ax[0].set_xlabel('1/λ')
        ax[0].set_ylabel("Rules in ensemble")
        ###
        ax[1].set_title('Impact of Regularization\non Accuracy', wrap=True)
        ax[1].plot(alphas, train_acc, label='Train accuracy')
        ax[1].plot(alphas, test_acc, label='Test accuracy')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('1/λ')
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        ###
        ax[2].set_title("Relationship Between Ensemble Size\nand Accuracy", wrap=True)
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy'], label='Old rules, Train dataset', c='b')
        ax[2].plot(range(self.n_rules + 1), self.history['accuracy_test'], label='Old rules, Test dataset', c='b',
                   linestyle='dashed')
        ax[2].plot(active_rule_number, train_acc, label='New rules, Train dataset', c='r')
        ax[2].plot(active_rule_number, test_acc, label='New rules, Test dataset', c='r', linestyle='dashed')
        ax[2].scatter(active_rule_number, train_acc, c='r')
        ax[2].scatter(active_rule_number, test_acc, c='r')
        ax[2].legend()
        ax[2].set_xlabel('Rules in the ensemble')
        ax[2].set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(
            os.path.join('Plots',
                         'Pruning',
                         'Embedded',
                         f'Accuracy_while_pruning_Model_{self.dataset_name}_{self.n_rules}_nu_{self.nu}_sampling_{self.sampling}_use_gradient_{self.use_gradient}.png')
        )

        plt.show()
        return
