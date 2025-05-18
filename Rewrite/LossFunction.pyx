cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AbsoluteErrorFunction:
    def __cinit__(self):
        pass

    cpdef double compute_decision(self, list[int] covered_instances,
                                   list[float] value_of_f,
                                   np.ndarray[np.float64_t, ndim=1] y):
        cdef list values = []
        cdef Py_ssize_t i, n = len(covered_instances)
        for i in range(n):
            if covered_instances[i] >= 0:
                values.append(y[i] - value_of_f[i])
        values.sort()

        if len(values) % 2 == 1:
            return values[len(values) // 2]
        else:
            return (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2.0

    cpdef int get_first_derivative(self, double y, double y_hat):
        cdef double diff = y - y_hat
        if diff > 0:
            return -1
        elif diff == 0:
            return 0
        else:
            return 1

    cpdef int get_second_derivative(self):
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class SquaredErrorFunction:
    def __cinit__(self):
        pass

    cpdef double compute_decision(self, list[int] covered_instances,
                                   list[float] value_of_f,
                                   np.ndarray[np.float64_t, ndim=1] y):
        cdef double decision = 0.0
        cdef int count = 0
        cdef Py_ssize_t i, n = len(covered_instances)

        for i in range(n):
            if covered_instances[i] == 1:
                decision += y[i] - value_of_f[i]
                count += 1

        if count == 0:
            return 0.0  # Avoid division by zero

        return decision / count

    cpdef double get_first_derivative(self, double y, double y_hat):
        return -2.0 * (y - y_hat)

    cpdef int get_second_derivative(self):
        return 2

# Could use typed memoryviews for better performance