# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
from libc.math cimport fabs, pow, floor, log10
cimport cython
from pyscipopt.scip cimport *
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer
from cpython cimport array
import array
cimport numpy as np
from time import time


cdef double[:] normalized_histogram(double[:] input_array, double low, double high, int steps):
    cdef double[:] bounds = np.linspace(low,high,steps+1)
    cdef int num_bins = bounds.shape[0] - 1
    cdef int[:] histogram = np.zeros(num_bins, dtype=np.int32)
    cdef int i, bin_idx
    cdef int bsz = bounds.shape[0]-1

    for i in range(input_array.shape[0]):
        for j in range(bsz):
            if bounds[j] < input_array[i]:
                if input_array[i] <= bounds[j+1]:
                    histogram[j]+=1
                    break

    return histogram / (np.sum(histogram)+1e-3)

    

cdef double sign(double x) nogil:
    if x >0:
        return 1
    elif x<0:
        return -1
    else:
        return 0
# Define the Cython version of signed_log function
cdef double[:] signed_log(double[:] x):
    return np.sign(x)*np.log(np.abs(x)+1e-3)

# Define the Cython version of powernorm function
cdef double powernorm(double val , double power) nogil:
    return sign(val) * (pow(fabs(val),power))


# Define the Cython version of make_data function
cdef make_data(double[:] vars, ):
    vars = np.abs(vars - np.floor(vars))
    #slack_cons = np.append(slack_cons, 0)

    #cdef double[:] slack_cons_filtered = signed_log(slack_cons)
    #cdef double mn = min(slack_cons_filtered)
    #cdef double mx = max(slack_cons_filtered) + 1e-8
    # range=(0,1.0), no range
    #cdef double[:] slack_hist = np.histogram(slack_cons_filtered,10, range=(mn,mx))[0].astype(np.double)
    #slack_hist = slack_hist / (np.sum(slack_hist) + 1e-8)

    cdef double frac_mean = np.mean(vars)
    cdef double[:] var_hist = np.histogram(vars, 10,range=(0,1.0))[0].astype(np.double)
    var_hist = var_hist / (np.sum(var_hist) + 1e-8)
    cdef double already_integral = np.isclose(vars, 0).mean()
    return var_hist, frac_mean, already_integral

# Define the Cython version of get_model_info function
def get_model_info(model, double power=0.5):
    cdef double NcutsApp = model.getNCutsApplied() /model.getNConss()
    cdef double Nsepa = model.getNSepaRounds()
    cdef double gap = model.getGap()
    cdef SCIP* model_ptr = <SCIP*> PyCapsule_GetPointer(model.to_ptr(False), "scip")
    
    cdef double[:] vars_array = get_discretization_errors(model_ptr)
    #cdef double[:] slack_cons_array = get_slack_vars(model_ptr)

    # Calculate slack_hist, var_hist, frac_mean, and already_integral using make_data function
    cdef double[:] slack_hist, var_hist
    cdef double frac_mean, already_integral
    var_hist, frac_mean, already_integral = make_data(vars_array)
    
    #cdef double cond = log10(model.getCondition())
    
    cdef double lpi = model.lpiGetIterations()
    
    info = {
        "NcutsApp": NcutsApp,
        "Nsepa": Nsepa,
        "gap": gap,
        "lpi": lpi,
        #"cond": cond,
        "mean to integral": frac_mean,
        #"std to integral": frac_std,
        #"max to integral": frac_max,
        #"min to integral": frac_min,
        "already_integral": already_integral
    }
    return info, np.asarray(var_hist), None
    
cdef get_discretization_errors(SCIP* model_scip, transformed=False):
    cdef SCIP_VAR** _vars
    cdef SCIP_VAR* _var
    cdef int _nvars
    cdef double val

    if transformed:
        _vars = SCIPgetVars(model_scip)
        _nvars = SCIPgetNVars(model_scip)
    else:
        _vars = SCIPgetOrigVars(model_scip)
        _nvars = SCIPgetNOrigVars(model_scip)

    
    cdef double[:] dc_error = np.empty(_nvars, dtype=np.double) 
    cdef int j = 0
    for i in range(_nvars):
        if SCIPvarGetType(_vars[i]) == SCIP_VARTYPE_CONTINUOUS:
            continue
        val = SCIPvarGetLPSol(_vars[i])
        dc_error[j] = val - floor(val)
        j+=1
    return dc_error[:j]

cdef get_slack_vars(SCIP* model_scip,):
    #model.getSlack(c) for c in model.getConss() if c.isOriginal() and c.isActive() and c.isLinear()
    cdef SCIP_CONS** _conss
    cdef int _nconss
    cdef double lhs, rhs, activity
    _conss = SCIPgetConss(model_scip)
    _nconss = SCIPgetNConss(model_scip)
    cdef double[:] conss = np.empty(_nconss, dtype=np.double)
    cdef int j = 0
    
    for i in range(_nconss):
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(_conss[i]))).decode("UTF-8")
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(model_scip, _conss[i])
            rhs = SCIPgetRhsLinear(model_scip, _conss[i])
            activity = SCIPgetActivityLinear(model_scip, _conss[i], NULL)
            lhsslack = activity - lhs
            rhsslack = rhs - activity
            conss[j] = min(lhsslack, rhsslack)
            j+=1

    return conss[:j]

