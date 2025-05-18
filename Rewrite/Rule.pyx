import numpy as np
cimport numpy as np

cdef double INF = 1e308  # A practical representation of "infinity" for doubles

cdef class Rule:
    cdef public object decision
    cdef public list conditions      # Each condition: [int attribute, float lower, float upper]
    cdef public list attribute_names

    def __cinit__(self):
        self.decision = None
        self.conditions = []
        self.attribute_names = []

    cpdef void add_condition(self, int best_attribute, double cut_value, int cut_direction, str attribute_name):
        cdef int GREATER_EQUAL = 1
        cdef list condition
        cdef int attr
        cdef double lower, upper

        for condition in self.conditions:
            attr = <int>condition[0]
            if attr == best_attribute:
                if cut_direction == GREATER_EQUAL:
                    condition[1] = max(cut_value, <double>condition[1])
                else:
                    condition[2] = min(cut_value, <double>condition[2])
                return

        # Create a new condition [attribute, lower_bound, upper_bound]
        condition = [None, None, None]
        condition[0] = best_attribute
        if cut_direction == GREATER_EQUAL:
            condition[1] = cut_value
            condition[2] = INF
        else:
            condition[1] = -INF
            condition[2] = cut_value
        self.conditions.append(condition)
        self.attribute_names.append(attribute_name)

    cpdef classify_instance(self, np.ndarray x, bint prune=False):
        cdef list condition
        cdef int attr
        cdef double lower, upper

        for condition in self.conditions:
            attr = <int>condition[0]
            lower = <double>condition[1]
            upper = <double>condition[2]
            if not lower <= x[attr] <= upper:
                if isinstance(self.decision, np.float64):
                    return 0
                else:
                    if type(self.decision) == list:
                        return [0 for _ in range(len(self.decision))]
                    return self.decision  # Added return statement for non-list decision
        return self.decision
