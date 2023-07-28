cimport cython
import numpy as np


# Define the Cython version of signed_log function
cdef double[:] signed_log(double[:] x):
    return np.sign(x)*np.log(np.abs(x)+1e-3)

# Define the Cython version of powernorm function
cdef double powernorm(double val , double power):
    return np.sign(val) * (np.abs(val)**power)

# Define the Cython version of make_data function
cdef make_data(double[:] vars, double[:] slack_cons):
    cdef double[:] vars_copy = np.abs(vars - np.floor(vars))
    slack_cons = np.append(slack_cons, 0)

    cdef double[:] slack_cons_filtered = signed_log(slack_cons)
    cdef double mn = max(min(slack_cons_filtered),-10)
    cdef double mx = min(max(slack_cons_filtered) + 1e-8,10)
    # range=(0,1.0), no range
    cdef double[:] slack_hist = np.histogram(slack_cons_filtered, 10, range=(mn,mx))[0].astype(np.double)
    slack_hist = slack_hist / (np.sum(slack_hist) + 1e-8)
    cdef double frac_mean = np.mean(vars_copy)
    cdef double[:] hist = np.histogram(vars_copy, 10, range=(0, 1.0))[0].astype(np.double)
    cdef double[:] var_hist = hist / (np.sum(hist) + 1e-8)
    cdef double already_integral = np.isclose(vars_copy, 0).mean()
    return slack_hist, var_hist, frac_mean, already_integral

# Define the Cython version of get_model_info function
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def get_model_info(model, double power=0.5):
    cdef double NcutsApp = powernorm(model.getNCutsApplied(),power)
    cdef double Nsepa = powernorm(model.getNSepaRounds(),power)
    cdef double gap = model.getGap()
    
    cdef double[:] vars_array = np.fromiter((v.getLPSol() for v in model.getVars() if v.vtype() in ["BINARY", "INTEGER", "IMPLINT"]),dtype=np.double)
    cdef double[:] slack_cons_array = np.fromiter((model.getSlack(c) for c in model.getConss() if c.isOriginal() and c.isActive() and c.isLinear()),dtype=np.double)


    # Calculate slack_hist, var_hist, frac_mean, and already_integral using make_data function
    cdef double[:] slack_hist, var_hist
    cdef double frac_mean, already_integral
    slack_hist, var_hist, frac_mean, already_integral = make_data(vars_array, slack_cons_array)

    cdef double cond = np.log10(model.getCondition())
    cdef double lpi = powernorm(model.lpiGetIterations(), power)

    info = {
        "NcutsApp": NcutsApp,
        "Nsepa": Nsepa,
        "gap": gap,
        "lpi": lpi,
        "cond": cond,
        "mean to integral": frac_mean,
        #"std to integral": frac_std,
        #"max to integral": frac_max,
        #"min to integral": frac_min,
        "already_integral": already_integral
    }
    return info, np.asarray(var_hist), np.asarray(slack_hist)


def get_discretization_errors(self, transformed):
    cdef SCIP_VAR** _vars
    cdef SCIP_VAR* _var
    cdef int _nvars
    dc_error = []

    if transformed:
        _vars = SCIPgetVars(self._scip)
        _nvars = SCIPgetNVars(self._scip)
    else:
        _vars = SCIPgetOrigVars(self._scip)
        _nvars = SCIPgetNOrigVars(self._scip)

    for i in range(_nvars):
        vartype = SCIPvarGetType(_vars[i])
        if vartype == SCIP_VARTYPE_CONTINUOUS:
            continue
        cdef double val = SCIPvarGetLPSol(_vars[i])
        dc_error.append(val - floor(val))
    return vars

def get_slack_vars(self,):
    #model.getSlack(c) for c in model.getConss() if c.isOriginal() and c.isActive() and c.isLinear()
    cdef SCIP_CONS** _conss
    cdef int _nconss
    conss = []

    _conss = SCIPgetConss(self._scip)
    _nconss = SCIPgetNConss(self._scip)
    for i in range(_nconss):
        constype = bytes(SCIPconshdlrGetName(SCIPconsGetHdlr(_conss[i]))).decode('UTF-8')
        if constype == 'linear':
            lhs = SCIPgetLhsLinear(self._scip, _conss[i])
            rhs = SCIPgetRhsLinear(self._scip, _conss[i])
            activity = SCIPgetActivityLinear(self._scip, _conss[i], NULL)
            lhsslack = activity - lhs
            rhsslack = rhs - activity
            conss.append(min(lhsslack, rhsslack))

    return conss
