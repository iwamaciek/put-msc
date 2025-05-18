from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error
from RiskMinimizer import AbsoluteErrorRiskMinimizer, GradientEmpiricalRiskMinimizer
from LossFunction import AbsoluteErrorFunction, SquaredErrorFunction
from Rule import Rule
from Cut import Cut
import matplotlib.pyplot as plt
import numpy as np
import cython

from sklearn.metrics import mean_absolute_error


class EnderRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_rules: int = 100, loss: str = 'squared_error_loss_function',
                 empirical_risk_minimizer: str = 'gradient_empirical_risk_minimizer', verbose: bool = False):
        self.verbose: bool = verbose
        self.n_rules: int = n_rules
        if loss == "squared_error_loss_function":
            self.loss_f: function = SquaredErrorFunction()
        elif loss == 'absolute_error_loss_function':
            self.loss_f: function = AbsoluteErrorFunction()

        if empirical_risk_minimizer == 'absolute_error_risk_minimizer':
            self.empirical_risk_minimizer = AbsoluteErrorRiskMinimizer(self.loss_f)
        elif empirical_risk_minimizer == "gradient_empirical_risk_minimizer":
            self.empirical_risk_minimizer = GradientEmpiricalRiskMinimizer(self.loss_f)

        self.value_of_f: list[float] = None
        self.rules: list[Rule] = None
        self.is_fitted_: bool = False

        self.nu: float = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.attribute_names: list[str] = X.columns
        X, y = check_X_y(X, y)
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.X = self.X.astype(np.float64)
        self.y = self.y.astype(np.float64)
        # self.loss = None

        self.value_of_f: list[float] = [0 for _ in range(len(X))]
        self.rules: list[Rule] = self.create_rules(self.X, self.y)

        self.is_fitted_ = True

        # return self

    def create_rules(self, X: np.ndarray, y: np.ndarray) -> list[Rule]:
        self.empirical_risk_minimizer.initialize_start(X, y)
        self.create_inverted_list(X)

        self.covered_instances: list[int] = [1 for _ in range(len(X))]
        rules: list[Rule] = [self.loss_f.compute_decision(self.covered_instances, self.value_of_f, y)]
        self.update_value_of_f(rules[0])
        if self.verbose:
            print('default_value (rule):', rules)
        for i_rule in range(self.n_rules):
            if self.verbose:
                print('####################################################################################')
                print(f"Rule: {i_rule + 1}")
            self.covered_instances: list[int] = [1 for _ in range(len(X))]

            rule: Rule = self.create_rule()

            if rule:
                self.update_value_of_f(rule.decision)
                rules.append(rule)
        return rules

    def create_rule(self) -> Rule:
        self.empirical_risk_minimizer.initialize_for_rule(self.value_of_f, self.covered_instances)

        rule: Rule = Rule()
        best_cut: Cut = Cut()
        best_cut.empirical_risk = 0

        creating: bool = True
        EPSILON: float = 1e-8
        while creating:
            best_attribute: int = -1
            cut: Cut = Cut()
            for attribute in range(len(self.X[0])):
                cut = self.find_best_cut(attribute)
                if cut.empirical_risk < best_cut.empirical_risk - EPSILON:
                    best_cut: Cut = cut
                    best_attribute: int = attribute
            if best_attribute == -1 or best_cut.exists is False:
                creating = False
            else:
                rule.add_condition(best_attribute, best_cut.value, best_cut.direction,
                                   self.attribute_names[best_attribute])
                self.mark_covered_instances(best_attribute, best_cut)

        if best_cut.exists:
            decision: float = self.nu * self.loss_f.compute_decision(self.covered_instances, self.value_of_f, self.y)
            rule.decision = decision

            if self.verbose:
                for i_condition in range(len(rule.conditions)):
                    if rule.conditions[i_condition][1] == -1e308:
                        print(f'\t{rule.attribute_names[i_condition]} <= {rule.conditions[i_condition][2]}')
                    elif rule.conditions[i_condition][2] == 1e308:
                        print(f'\t{rule.attribute_names[i_condition]} >= {rule.conditions[i_condition][1]}')
                    else:
                        print(
                            f'\t{rule.attribute_names[i_condition]} in [{rule.conditions[i_condition][1]}, {rule.conditions[i_condition][2]}]')

                print(f'=> Decision {rule.decision}')
                print()

            return rule
        else:
            return None

    def find_best_cut(self, attribute: int) -> Cut:
        best_cut: Cut = Cut()
        best_cut.position = -1
        best_cut.exists = False
        best_cut.empirical_risk = 0

        temp_empirical_risk: float = 0.0

        GREATER_EQUAL: int = 1
        LESS_EQUAL: int = -1
        EPSILON: float = 1e-8
        for cut_direction in [-1, 1]:
            self.empirical_risk_minimizer.initialize_for_cut()
            i: int = len(self.X) - 1 if cut_direction == GREATER_EQUAL else 0

            while (cut_direction == GREATER_EQUAL and i >= 0) or (
                    cut_direction != GREATER_EQUAL and attribute < len(self.X)):
                current_position: np.int64 = self.inverted_list[attribute, i]
                if self.covered_instances[current_position] > 0:
                    break
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1

            current_value = self.X[current_position][attribute]
            count: int = 0
            while (cut_direction == GREATER_EQUAL and i >= 0) or (cut_direction != GREATER_EQUAL and i < len(self.X)):
                next_position: np.int64 = self.inverted_list[attribute, i]
                if self.covered_instances[next_position] == 1:
                    if True:  # TODO ismissing(attribute)
                        count += 1
                        value: float = self.X[next_position][attribute]
                        weight: float = 1.0
                        if current_value != value and count >= 10:
                            if temp_empirical_risk < best_cut.empirical_risk - EPSILON:
                                best_cut.direction = cut_direction
                                best_cut.value = (current_value + value) / 2
                                best_cut.empirical_risk = temp_empirical_risk
                                best_cut.exists = True

                        temp_empirical_risk: float = self.empirical_risk_minimizer.compute_current_empirical_risk(
                            next_position, self.covered_instances[next_position] * weight)

                        current_value = self.X[next_position][attribute]
                i = i - 1 if cut_direction == GREATER_EQUAL else i + 1
        return best_cut

    def mark_covered_instances(self, best_attribute: int, cut: Cut):
        for i in range(len(self.X)):
            value = self.X[i][best_attribute]
            if (value < cut.value and cut.direction == 1) or (value > cut.value and cut.direction == -1):
                self.covered_instances[i] = -1

    def update_value_of_f(self, decision: float):
        for i in range(len(self.covered_instances)):
            if self.covered_instances[i] >= 0:
                self.value_of_f[i] += decision

    def create_inverted_list(self, X: np.ndarray):
        # import numpy as np
        # X = np.array(X)
        sorted_indices: np.ndarray[int] = np.argsort(X, axis=0)
        self.inverted_list: np.ndarray[int] = sorted_indices.T

    def predict(self, X: np.ndarray) -> list[float]:
        check_is_fitted(self, 'is_fitted_')

        X = check_array(X)

        predictions: list[float] = [self.predict_instance(x) for x in X]
        return predictions

    def predict_instance(self, x: np.ndarray) -> float:
        value_of_f_instance = self.rules[0]
        for rule in self.rules[1:]:
            value_of_f_instance += rule.classify_instance(x)
        return value_of_f_instance

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')

        X, y = check_X_y(X, y)

        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)

        return -mse

    def create_history(self, X_train, X_test, y_train, y_test):
        train_preds = [self.rules[0] for _ in range(len(X_train))]
        test_preds = [self.rules[0] for _ in range(len(X_test))]

        train_mae = [mean_absolute_error(y_train, train_preds)]
        test_mae = [mean_absolute_error(y_test, test_preds)]
        for rule in self.rules[1:]:
            for i_example, example in enumerate(X_train):
                train_preds[i_example] += rule.classify_instance(example)
            train_mae.append(mean_absolute_error(train_preds, y_train))
            for i_example, example in enumerate(X_test):
                test_preds[i_example] += rule.classify_instance(example)
            test_mae.append(mean_absolute_error(test_preds, y_test))
        return train_mae, test_mae

    def prune_rules(self, X_train, X_test, y_train, y_test):
        from sklearn.linear_model import lars_path
        import os

        rule_feature_matrix_train = [[0 if rule.classify_instance(x) == 0 else 1 for rule in self.rules[1:]] for x in
                                     X_train]
        rule_feature_matrix_test = [[0 if rule.classify_instance(x) == 0 else 1 for rule in self.rules[1:]] for x in
                                    X_test]

        lars_show_path = True

        _, _, coefs = lars_path(np.array(rule_feature_matrix_train, dtype=np.float64), np.array(y_train),
                                method="lasso", eps=1, verbose=True,
                                # positive=True
                                )

        print(coefs.T)
        if lars_show_path:
            xx = np.sum(np.abs(coefs.T), axis=1)
            xx /= xx[-1]

            plt.figure(figsize=(10, 7))
            plt.style.use('ggplot')
            plt.plot(xx, coefs.T, label=["Feature " + str(i+1) for i in range(len(coefs))])
            ymin, ymax = plt.ylim()
            # plt.vlines(xx, ymin, ymax, linestyle="dashed")
            plt.xlabel("|coef| / max|coef|")
            plt.ylabel("Coefficients")
            plt.title(f"LARS Path for Dataset Medical Cost with {self.n_rules} Attributes (Rules)")
            plt.axis("tight")
            plt.tight_layout()
            plt.legend()
            plt.savefig(
                os.path.join('Plots',
                             'pruning',
                             'Embedded',
                             f'Lars_path_Model_{self.dataset_name}_{self.n_rules}.png'))
            plt.show()
        rule_order = []
        matrix = coefs.T
        previous_row = np.zeros(matrix.shape[1])

        # Iterujemy przez każdy wiersz w macierzy
        for current_row in matrix:
            # Porównujemy z poprzednim wierszem
            for idx, (prev, curr) in enumerate(zip(previous_row, current_row)):
                if prev == 0 and curr != 0:
                    rule_order.append(idx+1)

            # Aktualizujemy poprzedni wiersz
            previous_row = current_row

        train_mae = []
        test_mae = []
        for c in coefs.T:
            train_preds = [self.rules[0] for _ in range(len(X_train))]
            for i_example, example in enumerate(rule_feature_matrix_train):
                train_preds[i_example] = sum(np.array(example) * np.array(c))/1
            train_mae.append(mean_absolute_error(y_train, train_preds))
            test_preds = [self.rules[0] for _ in range(len(X_test))]
            for i_example, example in enumerate(rule_feature_matrix_test):
                test_preds[i_example] = sum(np.array(example) * np.array(c))/1
            test_mae.append(mean_absolute_error(y_test, test_preds))
        print(train_mae)
        train_mae_hist, test_mae_hist = self.create_history(X_train, X_test, y_train, y_test)
        print(train_mae)
        print(test_mae)
        print(train_mae_hist)
        print(test_mae_hist)

        plt.plot(list(range(len(train_mae))), train_mae, c='y')
        plt.plot(list(range(len(train_mae))), train_mae_hist, c='b')
        plt.plot(list(range(len(test_mae))), test_mae, c='y', linestyle='dashed')
        plt.plot(list(range(len(test_mae))), test_mae_hist, c='b', linestyle='dashed')
        plt.show()


        print(rule_order)
        preds_train = [self.rules[0] for _ in range(len(X_train))]
        preds_test = [self.rules[0] for _ in range(len(X_test))]
        return