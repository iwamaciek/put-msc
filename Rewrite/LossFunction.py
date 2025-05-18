class AbsoluteErrorFunction():
    def __init__(self):
        pass

    def compute_decision(self, covered_instances, value_of_f, y):
        values = []
        for i in range(len(covered_instances)):
            if covered_instances[i] >= 0:
                values.append(y[i] - value_of_f[i])
        values = sorted(values)
        # print(values)
        if len(values) % 2 == 1:
            return values[len(values)//2]
        else:
            return sum(values[len(values)//2 - 1:len(values)//2 + 1])/2

    def get_first_derivative(self, y, y_hat):
        if y - y_hat > 0:
            return -1
        elif y - y_hat == 0:
            return 0
        elif y - y_hat < 0:
            return 1

    def get_second_derivative(self):
        return 0


class SquaredErrorFunction():
    def __init__(self):
        pass

    def compute_decision(self, covered_instances, value_of_f, y):
        decision = 0
        count = 0
        for i in range(len(covered_instances)):
            if covered_instances[i] == 1:
                decision += y[i] - value_of_f[i]
                count += 1
        return decision / count

    def get_first_derivative(self, y, y_hat):
        return -2*(y - y_hat)

    def get_second_derivative(self):
        return 2
