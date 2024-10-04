# online_transform.pxd

from abc import abstractmethod
cimport numpy as np

cdef class OnlineTransform:
    cdef public str input_feature_name
    cdef public double alpha
    cdef public object init_value
    cdef public bint adjust
    cdef public int n
    cdef public int required_n_warmup
    cdef public object value

    cpdef void update(self, double new_value)
    cpdef np.ndarray apply(self, np.ndarray data)
    cpdef object warm_value(self)


cdef class ExponentialMA(OnlineTransform):
    cdef public double numerator
    cdef public double denominator

    cpdef void update(self, double value)


cdef class ExponentialSTD(OnlineTransform):
    cdef public double s
    cdef public double s2
    cdef public double W
    cdef public double W2

    cpdef void update(self, double value)
