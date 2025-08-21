# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from libc.math cimport log
import time
from collections import OrderedDict
from cython.parallel cimport prange
import cython
cimport cython
import numpy as np
cimport numpy as np
# from numpy cimport ndarray
# from libc.math cimport sin
# from libc.math cimport fabs
# import math, time
# from cython.view cimport array as cvarray

LU_DTYPE = np.uint8
# cdef char[:] natural_lucodes = np.array([50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121,
#                    122, 130, 150, 151, 152, 153, 180], dtype=np.uint8)

NAT_MIN = 50
NAT_MAX = 180
natural_lucodes = [50, 60, 61, 62, 70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121,
                   122, 130, 140, 150, 152, 153, 160, 170, 180]


cdef int is_natural_lucode(int x):
    return (x >= 50) and (x <= 180)


@cython.boundscheck(False)
@cython.wraparound(False)
def restoration_cython(
    unsigned char[:, :] baselu,
    unsigned char[:, :] potveg,
    long[:, :] succession,
    dict code_to_sm_idx,
    dict code_to_base_code,
    unsigned char lu_nodata):

    cdef Py_ssize_t rows = baselu.shape[0]
    cdef Py_ssize_t cols = baselu.shape[1]

    result = np.empty((rows, cols), dtype=LU_DTYPE)
    cdef unsigned char[:, :] result_view = result

    cdef Py_ssize_t r, c
    cdef unsigned char lu, pv

    for r in range(rows):
        for c in range(cols):
            lu = code_to_base_code[baselu[r,c]]  # get rid of grazing/forestry flag
            if lu == lu_nodata:
                result_view[r,c] = lu
            elif (lu <= 40) or ((lu >= 200) and (lu <= 202)):
                result_view[r,c] = potveg[r,c]
            elif (lu < 190):
                pv = potveg[r,c]
                if (pv >= 50) and (pv < 190) and (succession[code_to_sm_idx[lu], code_to_sm_idx[pv]] == 1):
                    result_view[r,c] = pv
                else:
                    result_view[r,c] = lu
            else: 
                result_view[r,c] = lu
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def pasture_extensification_cython(
    unsigned char[:, :] base_lu,
    unsigned char[:, :] pot_veg,
    double [:, :] past_returns,
    double past_nodata):
    # """
    # Note that currently, there is no separate binary pasture suitability map. Instead, we use the 
    # potential pasture returns map, which has `past_nodata` where pasture is unsuitable, and some returns
    # value where it is suitable for pasture. 
    # """

    cdef Py_ssize_t rows = base_lu.shape[0]
    cdef Py_ssize_t cols = base_lu.shape[1]

    result = np.empty((rows, cols), dtype=LU_DTYPE)
    cdef unsigned char[:, :] result_view = result

    cdef Py_ssize_t r, c
    cdef unsigned char lu, pv

    for r in range(rows):
        for c in range(cols):
            lu = base_lu[r,c]
            pv = pot_veg[r,c]
            if (past_returns[r,c] == past_nodata) or (lu == 0) or (lu == 190) or (lu == 210) or (lu == 220):
                # unsuitable, fixed, or nodata land use (i.e. outside country extent)
                result_view[r, c] = lu
            else:
                # default case - convert to 134
                if (pv <= 130) or (pv >= 160):
                    result_view[r, c] = 134
                # special case for classes 140, 150, 152, 153
                else:
                    result_view[r, c] = pv + 4
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def forestry_extensification_cython(
    unsigned char[:, :] base_lu,
    unsigned char[:, :] pot_veg,
    double [:, :] forestry_returns):
    # """
    # INCOMPLETE
    # """

    cdef Py_ssize_t rows = base_lu.shape[0]
    cdef Py_ssize_t cols = base_lu.shape[1]

    result = np.empty((rows, cols), dtype=LU_DTYPE)
    cdef unsigned char[:, :] result_view = result

    cdef Py_ssize_t r, c
    cdef unsigned char lu, pv

    for r in range(rows):
        for c in range(cols):
            lu = base_lu[r,c]
            pv = pot_veg[r,c]
   
    return result


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def ext_curr_local_op(
    unsigned char [:, :] land_use,
    unsigned char [:, :] slope_threshold,
    unsigned char [:, :] rainfed_suitability,
    unsigned char [:, :] soil_suitability):

    cdef Py_ssize_t rows = land_use.shape[0]
    cdef Py_ssize_t cols = land_use.shape[1]

    result = np.empty((rows, cols), dtype=LU_DTYPE)
    cdef unsigned char[:, :] result_view = result

    cdef Py_ssize_t r, c

    for r in range(rows):
        for c in range(cols):
            if ((slope_threshold[r, c] == 1) and 
                (rainfed_suitability[r, c] == 1) and
                (soil_suitability[r,c] == 1) and
                is_natural_lucode(land_use[r, c])):
                result_view[r, c] = 10
            else:
                result_view[r, c] = land_use[r,c]
    
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def ext_irr_bmp_local_op(
    unsigned char [:, :] lu,
    unsigned char [:, :] st,
    unsigned char [:, :] rfs,
    unsigned char [:, :] irs,
    unsigned char [:, :] si,
    unsigned char [:, :] ss,
    unsigned char [:, :] rb,
    unsigned char [:, :] pv):

    cdef Py_ssize_t rows = lu.shape[0]
    cdef Py_ssize_t cols = lu.shape[1]

    result = np.empty((rows, cols), dtype=LU_DTYPE)
    cdef unsigned char[:, :] result_view = result

    cdef Py_ssize_t r, c

    for r in range(rows):
        for c in range(cols):
            # first check whether we're in a riparian buffer
            if (rb[r,c]==1) and (lu[r,c] <= 40):
                result_view[r,c] = pv[r,c]
            elif (lu[r,c] <= 180) and (st[r,c]==1) and (ss[r,c]==1):
                if si[r,c]==1: 
                    if irs[r,c]==1:
                        result_view[r,c] = 26
                    else:
                        result_view[r,c] = lu[r,c]
                else:
                    if rfs[r,c]==1:
                        result_view[r,c] = 16
                    else:
                        result_view[r,c] = lu[r,c]
            else:
                result_view[r,c] = lu[r,c]

    return result


