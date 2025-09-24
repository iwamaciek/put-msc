import numpy

class Rule():
    def __init__(self):
        self.decision = None
        self.conditions = []
        self.attribute_names = []

    def add_condition(self, best_attribute, cut_value, cut_direction, attribute_name):
        GREATER_EQUAL = 1
        for condition in self.conditions:
            if condition[0] == best_attribute:
                if cut_direction == GREATER_EQUAL:
                    condition[1] = max(cut_value, condition[1])
                else:
                    condition[2] = min(cut_value, condition[2])
                return
        condition = [None for _ in range(3)]
        condition[0] = best_attribute

        if cut_direction == GREATER_EQUAL:
            condition[1] = cut_value
            condition[2] = 99999999999999999
        else:
            condition[1] = -99999999999999999
            condition[2] = cut_value
        self.conditions.append(condition)
        self.attribute_names.append(attribute_name)

    def classify_instance(self, x, prune=False):
        for condition in self.conditions:
            if not condition[1] <= x[condition[0]] <= condition[2]:
                return 0 if type(self.decision) is numpy.float64 else [0 for _ in range(len(self.decision))]
        return self.decision
