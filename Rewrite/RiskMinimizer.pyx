import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class AbsoluteErrorRiskMinimizer:
    cdef object loss_f
    cdef np.ndarray gradients
    cdef np.ndarray y
    cdef double sum_of_weights
    cdef int sum_of_zero_residuals

    def __cinit__(self, loss_f):
        self.loss_f = loss_f

    cpdef initialize_start(self, np.ndarray[np.float64_t, ndim=2] X,
                                  np.ndarray[np.float64_t, ndim=1] y):
        self.gradients = np.zeros(X.shape[0], dtype=np.float64)
        self.y = y

    cpdef initialize_for_rule(self, list[float] value_of_f, list[int] covered_instances):
        cdef np.float64_t [:,:] gradients = self.gradients
        cdef np.float64_t [:] y = self.y
        cdef Py_ssize_t i, n = len(covered_instances)
        for i in range(n):
            if covered_instances[i] == 1:
                gradients[i] = self.loss_f.get_first_derivative(y[i], value_of_f[i])

    cpdef initialize_for_cut(self):
        self.sum_of_weights = 0.0
        self.sum_of_zero_residuals = 0

    cpdef double compute_current_empirical_risk(self, Py_ssize_t position, double weight):
        self.sum_of_weights += weight * self.gradients[position]
        cdef np.float64_t [:] gradients = self.gradients
        if gradients[position] == 0.0:
            self.sum_of_zero_residuals += 1
        return -abs(self.sum_of_weights) + self.sum_of_zero_residuals


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class GradientEmpiricalRiskMinimizer:
    cdef object loss_f
    cdef np.ndarray gradients
    cdef np.ndarray y
    cdef double sum_of_weights
    cdef double count

    def __cinit__(self, loss_f):
        self.loss_f = loss_f

    cpdef initialize_start(self, np.ndarray[np.float64_t, ndim=2] X,
                                  np.ndarray[np.float64_t, ndim=1] y):
        self.gradients = np.zeros(X.shape[0], dtype=np.float64)
        self.y = y

    cpdef initialize_for_rule(self, list[float] value_of_f, list[int] covered_instances):
        cdef np.float64_t [:] gradients = self.gradients
        cdef np.float64_t [:] y = self.y
        cdef Py_ssize_t i, n = len(covered_instances)
        for i in range(n):
            if covered_instances[i] == 1:
                gradients[i] = self.loss_f.get_first_derivative(y[i], value_of_f[i])

    cpdef initialize_for_cut(self):
        self.sum_of_weights = 0.0
        self.count = 0.0

    cpdef double compute_current_empirical_risk(self, Py_ssize_t position, double weight):
        cdef np.float64_t [:] gradients = self.gradients
        self.sum_of_weights += weight * gradients[position]
        self.count += weight
        if self.count == 0:
            return 0.0
        return -abs(self.sum_of_weights) / self.count**0.5
