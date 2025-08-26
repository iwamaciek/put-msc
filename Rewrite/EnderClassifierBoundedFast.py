import math
import random
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import cython
import time


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
    def __init__(self, dataset_name = None, n_rules = 100, use_gradient = True, optimized_searching_for_cut = False, nu = 1,
                 sampling = 1, verbose = True, random_state = 42, max_clusters = 4, lambda_reg = 0.0):
        self.dataset_name: str = dataset_name
        self.n_rules: int = n_rules
        self.rules: list[Rule] = []

        self.use_gradient: bool = use_gradient
        self.nu: float = nu
        self.sampling: float = sampling
        self.max_clusters: int = max_clusters
        self.lambda_reg: float = lambda_reg

        self.verbose: bool = verbose
        self.random_state: int = random_state
        random.seed(random_state)

        self.optimized_searching_for_cut: bool = optimized_searching_for_cut
        self.history = {'accuracy': [],
                        'mean_absolute_error': [],
                        'accuracy_test': [],
                        'mean_absolute_error_test': []}

        self.is_fitted_: bool = False

        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_test: np.ndarray = None
        self.attribute_names: list[str] = None
        self.num_classes: int = None
        self.value_of_f: list[list[float]] = None
        self.probability: np.ndarray = None
        self.default_rule: np.ndarray = None
        self.covered_instances: list[int] = None
        self.instances_covered_by_current_rule: np.ndarray = None
        self.last_index_computation_of_empirical_risk: int = None
        self.gradient: np.ndarray = None
        self.hessian: np.ndarray = None
        self.gradients: list[float] = None
        self.hessians: list[float] = None
        self.inverted_list: np.ndarray[np.intc] = None
        self.indices_for_better_cuts: list[int] = None
        self.max_k: int = None
        self.effective_rules: list = None

        self.squared_error_array: np.ndarray = None

        plt.style.use('ggplot')

    def fit(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray = None, y_test: np.ndarray = None) -> None:
        global_start = time.perf_counter()
        self.rule_times = []
        self.rule_total_times = []
        self.attribute_names: list[str] = X.columns
        X, y = check_X_y(X, y, ensure_all_finite=False)
        if X_test is not None and y_test is not None:
            X_test, y_test = check_X_y(X_test, y_test, ensure_all_finite=False)
            self.X_test = X_test
            self.y_test = y_test
        self.X = X.astype(np.float64)
        self.y = y.astype(np.intc)

        self.num_classes: int = len(set(y))
        self.value_of_f: list[list[float]] = [[0 for _ in range(self.num_classes)] for _ in range(len(self.X))]
        self.probability: np.ndarray = np.zeros((len(self.X), self.num_classes), dtype=np.float64)

        self.create_rules(X)

        self.is_fitted_: bool = True
        total_time = time.perf_counter() - global_start
        if self.verbose:
            print(f"Total training time: {total_time:.5f} seconds")

        return self.rule_times, self.rule_total_times, total_time

    def create_rules(self, X: np.ndarray):
        self.create_inverted_list(X)
        self.covered_instances: np.ndarray[np.intc] = np.array([1 for _ in range(len(X))], dtype=np.intc)

        default_start = time.perf_counter()
        self.default_rule: np.ndarray = self.create_default_rule()
        if self.verbose:
            print(f"Default rule creation time: {time.perf_counter() - default_start:.5f} seconds")
        self.rules: list[Rule] = []
        self.update_value_of_f(self.default_rule)
        if self.verbose: print("Default rule:", self.default_rule)
        i_rule = 0
        start_time = time.perf_counter()
        while i_rule < self.n_rules:
            if self.verbose:
                print('####################################################################################')
                print(f"Rule: {i_rule + 1}")
            rule_start = time.perf_counter()
            self.squared_error_array = self.calculate_squared_errors()
            self.covered_instances: np.ndarray[np.intc] = self.resampling()
            rule: Rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                self.rules.append(rule)
                i_rule += 1
            rule_end = time.perf_counter()
            self.rule_times.append(rule_end - rule_start)
            self.rule_total_times.append(rule_end - start_time)
            if self.verbose:
                print(f"Rule {i_rule} creation time: {self.rule_times[-1]:.5f} seconds, total: {self.rule_total_times[-1]:.5f} seconds")
            # else:
            #     break

    def resampling(self) -> np.ndarray[np.intc]:
        count: Counter = Counter(self.y)
        total: int = len(self.y)
        no_examples_to_use: int = math.ceil(len(self.y) * self.sampling)

        ones_allocation: dict = {key: round((value / total) * no_examples_to_use) for key, value in count.items()}

        allocated_ones: int = sum(ones_allocation.values())
        difference: int = no_examples_to_use - allocated_ones
        keys: list = list(ones_allocation.keys())

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

        result: list[int] = [0] * len(self.y)

        for key, num_ones in ones_allocation.items():
            indices = [i for i, x in enumerate(self.y) if x == key]
            selected_indices = random.sample(indices, num_ones)
            for index in selected_indices:
                result[index] = 1
        return np.array(result, dtype=np.intc)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.locals(
        i=cython.int,
        j=cython.int,
        k=cython.int,
        w=cython.double,
        y_sum=cython.double,
        y2_sum=cython.double,
        mean=cython.double,
        n=cython.int,
        prefix_w_view=cython.double[:],
        prefix_y_view=cython.double[:],
        prefix_y2_view=cython.double[:],
        cost_view=cython.double[:,:],
        dp_view=cython.double[:,:],
    )
    def weighted_1d_kmeans(self, y_means: np.ndarray, weights: np.ndarray, max_clusters: int) -> np.float64:
        n = y_means.shape[0]

        prefix_w = np.zeros(n + 1, dtype=np.float64)
        prefix_y = np.zeros(n + 1, dtype=np.float64)
        prefix_y2 = np.zeros(n + 1, dtype=np.float64)

        prefix_w[1:] = np.cumsum(weights)
        prefix_y[1:] = np.cumsum(weights * y_means)
        prefix_y2[1:] = np.cumsum(weights * y_means ** 2)

        prefix_w_view = prefix_w
        prefix_y_view = prefix_y
        prefix_y2_view = prefix_y2

        dp = np.full((max_clusters + 1, n + 1), np.inf, dtype=np.float64)
        dp_view = dp

        cost = np.zeros((n, n), dtype=np.float64)
        cost_view = cost

        for i in range(n):
            for j in range(i, n):
                w = prefix_w_view[j + 1] - prefix_w_view[i]
                if w == 0:
                    cost_view[i, j] = 0
                else:
                    y_sum = prefix_y_view[j + 1] - prefix_y_view[i]
                    y2_sum = prefix_y2_view[j + 1] - prefix_y2_view[i]
                    mean = y_sum / w
                    cost_view[i, j] = y2_sum - 2 * mean * y_sum + w * mean * mean

        dp_view[0, 0] = 0.0
        for k in range(1, max_clusters + 1):
            for j in range(1, n + 1):
                for i in range(j):
                    dp_view[k, j] = min(dp_view[k, j], dp_view[k - 1, i] + cost_view[i, j - 1])

        return np.min(dp_view[1:max_clusters + 1, n])
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute_rule_lower_bound(self, X_subset: np.ndarray, residuals: np.ndarray, y_subset: np.ndarray, lambda_reg:float=0.0, max_clusters:int=4):
        """
        Compute the k-Means Equivalent Points Lower Bound for a partial rule's coverage.
        
        Parameters:
        - X_subset: (n_samples, n_features) samples covered by current rule
        - residuals: (n_samples,) residual targets (y - f(x))
        - lambda_reg: regularization parameter
        - max_clusters: maximum clusters to simulate (rule complexity)
        
        Returns:
        - lower_bound: float
        """
        X_view = X_subset.view([('', X_subset.dtype)] * X_subset.shape[1])
        _, inv, counts = np.unique(X_view, return_inverse=True, return_counts=True)

        num_groups = counts.shape[0]
        y_means = np.zeros(num_groups, dtype=np.float64)
        weights = counts.astype(np.float64)

        for i in range(len(residuals)):
            y_means[inv[i]] += residuals[i]

        y_means /= weights

        y_means_array: np.ndarray = np.array(y_means, dtype=np.float64)
        weights_array: np.ndarray = np.array(weights, dtype=np.float64)

        # Lower bound via k-means loss
        kmeans_loss: np.float64 = self.weighted_1d_kmeans(y_means_array, weights_array, max_clusters)

        # Correction: actual vs compressed squared sums
        original_total = np.sum(np.square(residuals))
        compressed_total = np.sum([w * ym ** 2 for w, ym in zip(weights, y_means)])
        correction = original_total - compressed_total

        return (kmeans_loss + lambda_reg * max_clusters + correction) / len(residuals)
    
    @cython.locals(
            squared_errors=cython.double[:,:],
            # logits=cython.double[:,:],
            i=cython.int,
            rule_id=cython.int,
            X_view=cython.double[:,:],
            xsize=cython.int,
            nrules=cython.int,
            y_view=cython.int[:],
            num_classes=cython.int,
            default_rule=cython.double[:,:],
            squared_errors_internal=cython.double[:],
            mses= cython.double[:],
    )
    def calculate_squared_errors(self) -> np.ndarray:
        """
        Iterates over all samples and all rules and calculates the squared errors for each rule.
        """
        def softmax(logits: np.ndarray) -> np.ndarray:
            e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return e / np.sum(e, axis=-1, keepdims=True)
        
        def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
            return np.eye(num_classes)[y]
        
        def mse_from_logits(logits: np.ndarray, y_true: np.ndarray, num_classes: int) -> np.ndarray:
            probs: np.ndarray = softmax(logits)  # shape: (n_samples, n_classes)
            y_onehot: np.ndarray = one_hot(y_true, num_classes)  # shape: (n_samples, n_classes)
            squared_errors_internal: np.ndarray = np.mean((probs - y_onehot) ** 2, axis=1)
            return squared_errors_internal

        X_view = self.X
        xsize = len(X_view)
        nrules = len(self.rules)
        num_classes = self.num_classes
        y_view = self.y
        default_rule = self.default_rule.reshape(1, -1)  # Reshape to (1, num_classes)

        squared_errors = np.zeros((nrules + 1, xsize), dtype=np.float64)
        mses = mse_from_logits(default_rule, y_view, num_classes)
        for i in range(xsize):
            squared_errors[0, i] = mses[i]
        if self.rules:
            for rule_id in range(nrules):
                logits = np.zeros((xsize, num_classes), dtype=np.float64)
                for i in range(xsize):
                    logits[i,:] = self.rules[rule_id].classify_instance(self.X[i])
                mses = mse_from_logits(logits, y_view, num_classes)
                for i in range(xsize):
                    squared_errors[rule_id + 1, i] = mses[i]
                rule_id += 1
        squared_errors = np.array(squared_errors, dtype=np.float64)
        if self.verbose:
            for i in range(len(squared_errors)):
                if i == 0:
                    print(f"Default Rule: MSE = {np.mean(squared_errors[i])}")
                else:
                    print(f"Rule {i-1}: MSE = {np.mean(squared_errors[i])}")
        squared_errors_numpy = np.array(squared_errors, dtype=np.float64)
        # self.squared_error_array = squared_errors_numpy
        return squared_errors_numpy


    def get_best_rule_loss(self, instances_covered_by_current_rule) -> float:
        """
        Iterates over all rules and finds which one has the lowest mean squared error loss.
        Returns:
        - best_rule_loss: float, the empirical risk of the best rule
        """
        best_rule_loss = np.inf
        a = self.squared_error_array[:, instances_covered_by_current_rule == 1]
        mses_l = np.mean(a, axis=1)
        best_rule_loss = np.min(mses_l)
        return best_rule_loss

    @cython.locals(
            inv_list=cython.int[:,:],
            attribute=cython.int,
            xsize=cython.int,
            n_attributes=cython.int,
            X_view=cython.double[:,:],
            covered_instances_view=cython.int[:],
            y_view=cython.int[:],
            max_k=cython.int,
            use_gradient=cython.bint,
            probability_view=cython.double[:,:],
            decisions=cython.int[:],
            positions=cython.int[:],
            directions=cython.int[:],
            values=cython.double[:],
            empirical_risks=cython.double[:],
            existss=cython.char[:],
            gradient=cython.double,
            abs_gradient=cython.double,
            hessian=cython.double,
            best_cut_decision=cython.int,
            best_cut_position=cython.int,
            best_cut_direction=cython.int,
            best_cut_value=cython.double,
            best_cut_empirical_risk=cython.double,
            best_cut_exists=cython.bint,
            temp_empirical_risk=cython.double,
            GREATER_EQUAL=cython.int,
            EPSILON=cython.double,
            previous_position=cython.int,
            curr_position=cython.int,
            previous_value=cython.double,
            cut_direction=cython.int,
            i=cython.int,
            j=cython.int,
            curr_value=cython.double,
            weight=cython.double,
            PRE_CHOSEN_K_L=cython.int,
            INSTANCE_WEIGHT_L=cython.double,
            Rp_L=cython.double,
            curr_best_attribute=cython.int,
    )
    @cython.boundscheck(False)
    def create_rule(self):
        self.initialize_for_rule()
        rule: Rule = Rule()

        best_cut: Cut = Cut()
        best_cut.empirical_risk = 0

        creating: bool = True
        EPSILON: float = 1e-8
        PRE_CHOSEN_K_L = PRE_CHOSEN_K
        INSTANCE_WEIGHT_L = INSTANCE_WEIGHT
        Rp_L = Rp
        # count: int = 0
        n_attributes: int = len(self.X[0])
        xsize = len(self.X)
        X_view = self.X
        y_view = self.y
        inv_list = self.inverted_list
        covered_instances_view = self.covered_instances
        probability_view = self.probability
        max_k = self.max_k
        use_gradient = self.use_gradient
        while creating:
            # count += 1
            best_attribute = -1
            decisions = np.zeros(n_attributes, dtype=np.intc)
            positions = np.zeros(n_attributes, dtype=np.intc)
            directions = np.zeros(n_attributes, dtype=np.intc)
            values = np.zeros(n_attributes, dtype=np.float64)
            empirical_risks = np.zeros(n_attributes, dtype=np.float64)
            existss = np.zeros(n_attributes, dtype=np.bool_)
            for attribute in range(n_attributes):#, nogil=True):
                best_cut_decision: np.intc = 0
                best_cut_position: np.intc = -1
                best_cut_direction: np.intc = 0
                best_cut_value: np.float64 = 0.0
                best_cut_empirical_risk: np.float64 = 0.0
                best_cut_exists: np.bool_ = False
                temp_empirical_risk: np.float64 = 0.0

                GREATER_EQUAL: np.intc = 1
                # LESS_EQUAL: np.intc = -1
                EPSILON: np.float64 = 1e-8

                for j in range(2):
                    if j == 0:
                        cut_direction = -1
                    else:
                        cut_direction = 1
                    gradient: np.float64 = 0.0
                    hessian: np.float64 = 5.0 #R

                    if cut_direction == GREATER_EQUAL:
                        i: np.intc = xsize - 1 
                    else:
                        i: np.intc = 0
                    previous_position = inv_list[attribute][i]
                    previous_value = X_view[previous_position][attribute]
                    while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < xsize):
                        curr_position = inv_list[attribute][i]
                        if covered_instances_view[curr_position] == 1:
                            if True:
                                curr_value = X_view[curr_position, attribute]
                                weight: np.float64 = 1.0

                                if previous_value != curr_value:
                                    if temp_empirical_risk < best_cut_empirical_risk - EPSILON:
                                        best_cut_direction = cut_direction
                                        best_cut_value = (previous_value + curr_value) / 2
                                        best_cut_empirical_risk = temp_empirical_risk
                                        best_cut_exists = True

                                if PRE_CHOSEN_K_L:
                                    if y_view[curr_position] == max_k:
                                        gradient += INSTANCE_WEIGHT_L * weight
                                    gradient = gradient - (INSTANCE_WEIGHT_L * weight * probability_view[curr_position, max_k])
                                    if use_gradient:
                                        temp_empirical_risk = -gradient
                                    else:
                                        hessian = hessian + (INSTANCE_WEIGHT_L * weight * (Rp_L + probability_view[curr_position, max_k] * (1 - probability_view[curr_position, max_k])))
                                        if gradient < 0:
                                            abs_gradient = -gradient
                                        else:
                                            abs_gradient = gradient
                                        temp_empirical_risk = -gradient * abs_gradient / hessian
                                previous_value = X_view[curr_position, attribute]
                        if cut_direction == GREATER_EQUAL:
                            i = i - 1
                        else:
                            i = i + 1
                decisions[attribute] = best_cut_decision
                positions[attribute] = best_cut_position
                directions[attribute] = best_cut_direction
                values[attribute] = best_cut_value
                empirical_risks[attribute] = best_cut_empirical_risk
                existss[attribute] = best_cut_exists

            for curr_best_attribute in range(n_attributes):
                if empirical_risks[curr_best_attribute] < best_cut.empirical_risk - EPSILON:
                    best_cut.decision = decisions[curr_best_attribute]
                    best_cut.position = positions[curr_best_attribute]
                    best_cut.direction = directions[curr_best_attribute]
                    best_cut.value = values[curr_best_attribute]
                    best_cut.empirical_risk = empirical_risks[curr_best_attribute]
                    best_cut.exists = existss[curr_best_attribute]
                    best_attribute = curr_best_attribute

            if best_attribute == -1 or not best_cut.exists:
                creating = False
            else:
                rule.add_condition(best_attribute, best_cut.value, best_cut.direction,
                                   self.attribute_names[best_attribute])
                self.mark_covered_instances(best_attribute, best_cut)
                # Verify lower bound for the empirical risk with more cuts
                # start = time()
                lower_bound = self.compute_rule_lower_bound(
                    X_subset=self.X[self.instances_covered_by_current_rule == 1],
                    residuals=[y_view[i] - self.value_of_f[i][max_k] for i in range(xsize) if
                               self.instances_covered_by_current_rule[i] == 1],
                    y_subset=self.y[self.instances_covered_by_current_rule == 1],
                    lambda_reg=self.lambda_reg, max_clusters=self.max_clusters)
                # if self.verbose:
                #     print(f"Lower bound computed in {time() - start:.4f} seconds, value: {lower_bound}")
                best_rule_loss = self.get_best_rule_loss(self.instances_covered_by_current_rule)
                if lower_bound < best_rule_loss:
                    if False:
                        print(f"Lower bound {lower_bound} is less than empirical risk {best_rule_loss} for rule with {len(rule.conditions)} conditions. Continuing search.")
                else:
                    if self.verbose and False:
                        print(f"Lower bound {lower_bound} is greater than or equal to empirical risk {best_rule_loss} for rule with {len(rule.conditions)} conditions. Stopping search.")
                    creating = False
        if best_cut.exists:
            decision = self.compute_decision()
            if decision is None:
                return None

            decision = [dec * self.nu for dec in decision]

            rule.decision = decision
            if self.verbose:
                for i_condition in range(len(rule.conditions)):
                    if rule.conditions[i_condition][1] == -np.inf:
                        print(f'\t{rule.attribute_names[i_condition]} <= {rule.conditions[i_condition][2]}')
                    elif rule.conditions[i_condition][2] == np.inf:
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
        
    def find_best_cut(self, attribute: int) -> Cut:
        best_cut = Cut()
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0

        temp_empirical_risk: float = 0

        GREATER_EQUAL: int = 1
        LESS_EQUAL: int = -1
        EPSILON: float = 1e-8

        empirical_risks: list[float] = []
        indices_to_check: list[int] = []
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

    def mark_covered_instances(self, best_attribute: int, cut: Cut):
        for i in range(len(self.X)):
            value = self.X[i][best_attribute]
            if (self.instances_covered_by_current_rule[i] == 1) and ((value >= cut.value and cut.direction == 1) or (value <= cut.value and cut.direction == -1)):
                self.instances_covered_by_current_rule[i] = 1
            else:
                self.instances_covered_by_current_rule[i] = 0
            if self.covered_instances[i] != -1:
                if (value < cut.value and cut.direction == 1) or (value > cut.value and cut.direction == -1):
                    self.covered_instances[i] = -1

    def initialize_for_cut(self):
        self.gradient = 0
        self.hessian = R
        self.gradients = [0 for _ in range(self.num_classes)]
        self.hessians = [R for _ in range(self.num_classes)]

    def compute_current_empirical_risk_optimized(self, next_position: np.int64, weight: float) -> tuple[float, float]:
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

    def compute_current_empirical_risk(self, next_position: np.int64, weight: float) -> float:
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

    def create_default_rule(self) -> np.ndarray:
        self.initialize_for_rule()
        decision: np.ndarray = self.compute_decision()
        for i in range(self.num_classes):
            decision[i] *= self.nu
        return decision

    def compute_decision(self) -> np.ndarray | None:
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
            decision = np.array(decision, dtype=np.float64)
            return decision
        else:
            raise

    def initialize_for_rule(self):
        self.instances_covered_by_current_rule = np.ones(len(self.X), dtype=int)

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
        X: np.ndarray = np.array(X)
        sorted_indices = np.argsort(X, axis=0).astype(np.intc)
        self.inverted_list = sorted_indices.T
        temp = self.inverted_list.copy()
        temp = np.array([[self.y[temp[i][j]] for j in range(len(temp[0]))] for i in range(len(temp))])

    def update_value_of_f(self, decision: np.ndarray) -> None:
        for i in range(len(self.X)):
            if self.covered_instances[i] >= 0:
                for k in range(self.num_classes):
                    self.value_of_f[i][k] += decision[k]

    def predict(self, X: np.ndarray, use_effective_rules: bool = True) -> list:
        X = check_array(X, ensure_all_finite=False)
        predictions = [self.predict_instance(x, use_effective_rules) for x in X]
        predictions = [np.argmax(pred) for pred in predictions]
        return predictions

    def predict_logits(self, X: np.ndarray, use_effective_rules: bool = True) -> np.ndarray:
        X = check_array(X, ensure_all_finite=False)
        logits = [self.predict_instance(x, use_effective_rules) for x in X]
        return np.array(logits)

    def predict_proba(self, X: np.ndarray, use_effective_rules: bool = True) -> np.ndarray:
        X = check_array(X, ensure_all_finite=False)
        predictions = self.predict_logits(X, use_effective_rules)
        exps = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
        probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        return probabilities

    def predict_instance(self, x: np.ndarray, use_effective_rules: bool) -> np.ndarray:
        value_of_f_instance = np.array(self.default_rule)
        rules = self.rules
        for rule in rules:
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance

    def predict_with_specific_rules(self, X: np.ndarray, rule_indices: list) -> np.ndarray:
        X = check_array(X, ensure_all_finite=False)
        preds = []
        for x in X:
            pred = np.array(self.default_rule)
            for rule_index in rule_indices:
                pred += np.array(self.rules[rule_index].classify_instance(x))
            preds.append(pred)
        return np.array(preds)

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')

        X, y = check_X_y(X, y, ensure_all_finite=False)

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
