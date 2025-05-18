class AbsoluteErrorRiskMinimizer():
    def __init__(self, loss_f):
        self.loss_f = loss_f

    def initialize_start(self, X, y):
        self.gradients = [0 for _ in range(len(X))]
        self.y = y

    def initialize_for_rule(self, value_of_f, covered_instances):
        for i in range(len(covered_instances)):
            if covered_instances[i] == 1:
                self.gradients[i] = self.loss_f.get_first_derivative(self.y[i], value_of_f[i])

    def initialize_for_cut(self):
        self.sum_of_weights = 0
        self.sum_of_zero_residuals = 0

    def compute_current_empirical_risk(self, position, weight):
        self.sum_of_weights += weight * self.gradients[position]
        if self.gradients[position] == 0: self.sum_of_zero_residuals += 1
        return -abs(self.sum_of_weights) + self.sum_of_zero_residuals


class GradientEmpiricalRiskMinimizer():
    def __init__(self, loss_f):
        self.loss_f = loss_f

    def initialize_start(self, X, y):
        self.gradients = [0 for _ in range(len(X))]
        self.y = y

    def initialize_for_rule(self, value_of_f, covered_instances):
        for i in range(len(covered_instances)):
            if covered_instances[i] == 1:
                self.gradients[i] = self.loss_f.get_first_derivative(self.y[i], value_of_f[i])

    def initialize_for_cut(self):
        self.sum_of_weights = 0
        self.count = 0

    def compute_current_empirical_risk(self, position, weight):
        self.sum_of_weights += weight * self.gradients[position]
        self.count += weight
        return -abs(self.sum_of_weights) / self.count ** .5
