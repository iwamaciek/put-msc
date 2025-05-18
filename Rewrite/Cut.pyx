cdef class Cut:
    cdef public int decision
    cdef public int position
    cdef public int direction
    cdef public double value
    cdef public double empirical_risk
    cdef public bint exists

    def __cinit__(self):
        self.decision = 0
        self.position = -1
        self.direction = 0
        self.value = 0.0
        self.empirical_risk = 0.0
        self.exists = False