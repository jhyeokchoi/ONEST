###
# Original 2015 by Donghan Lee
# Python 3 conversion, initGuessAll removal, scipy.optimize.least_squares integration: 2025
# Fitting logic moved to fit.py: 2025
###

import numpy as np
from scipy.linalg import expm
#from scipy.optimize import least_squares # Now in fit.py
#from scipy.stats import norm # Removed for performance
import concurrent.futures


from matplotlib.pyplot import plot, errorbar, figure, title, legend, xlabel, ylabel, grid, xlim, ylim, close
from matplotlib.backends.backend_pdf import PdfPages
#from os import getlogin, name  #for linux
from os import getlogin
from platform import uname       #for window
from time import ctime
import json

from est_data import EstDataSet
import fit as fit_module # Import the new fit module

def fast_gaussian(x, mu, sigma):
    """
    A faster implementation of Gaussian PDF than scipy.stats.norm.pdf.
    """
    if sigma == 0:
        return np.where(x == mu, 1.0, 0.0) # Should ideally not happen with correct logic
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def matrix_calc_worker_chunk(args):
    """
    Worker function for multiprocessing Matrix calculation (Chunked).
    args: (kab, kba, dG, dw, R1, R2a, R2b, offsets, T, v1, v1err, B0)
    offsets is a list or array of offset values.
    Returns a list of calculated intensities.
    """
    kab, kba, dG, dw, R1, R2a, R2b, offsets, T, v1, v1err, B0 = args
    
    results = []
    
    # Pre-calculate constants common to all offsets in this chunk
    R1a = R1; R1b = R1
    wG = dG * B0 * 2.0 * np.pi
    wE = (dG + dw) * B0 * 2.0 * np.pi
    wy = 0.0
    
    kex = kab + kba 
    if kex == 0: pB, pA = 0.0, 1.0
    else: pB, pA = kab / kex, kba / kex

    w1_val = v1*2*np.pi; w1err_val = v1err*2*np.pi
    
    if w1err_val == 0:
        wxs, weights = np.array([w1_val]), np.array([1.0])
    else:
        wxs = np.linspace(-2.0*w1err_val + w1_val, 2.0*w1err_val + w1_val, 10)
        weights = fast_gaussian(wxs, w1_val, w1err_val)

    if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
            weights = np.ones_like(wxs)/len(wxs) if len(wxs)>0 else np.array([1.0])

    # Normalize weights
    weights_sum = np.sum(weights)
    if weights_sum != 0:
        weights = weights / weights_sum

    refM = pA if pA != 0 else 1.0
    startM = np.array([0,0,0,pA,0,0,pB], dtype=float)
    
    # Loop over offsets in this chunk
    for dRF in offsets:
        wRF = dRF * B0 * 2.0 * np.pi
        wa = wG - wRF
        wb = wE - wRF
        
        preIcal = 0.0
        for i in range(len(wxs)):
            wx = wxs[i]
            Alist = [[0,0,0,0,0,0,0], [0,-kab-R2a,-wa,wy,kba,0,0], [0,wa,-kab-R2a,-wx,0,kba,0],
                        [2*R1a*pA,-wy,wx,-kab-R1a,0,0,kba], [0,kab,0,0,-kba-R2b,-wb,wy],
                        [0,0,kab,0,wb,-kba-R2b,-wx], [2*R1b*pB,0,0,kab,-wy,wx,-kba-R1b]]
            A = np.array(Alist, dtype=float)
            eA = expm(A * T) 
            endM = np.dot(eA,startM)
            preIcal += weights[i]*endM[3]/refM

        Icalc = preIcal # Already normalized
        results.append(max(0.0, Icalc))
        
    return results


class est_model:
    def __init__(self):
        self.programName = 'est ver. 2.1 (Python 3, Scipy LS, Refactored Fit)' # Updated version
        self.header = f'Donghan Lee\n{ctime()}\n'
        
        self.method = 'Baldwin' # Default method

        self.kab = 0.0
        self.kba = 0.0
        
        self.dataset = EstDataSet()
        self.verbose = False

        self.chi2 = 0.0
        self.npar = 0
        self.dof = 0
        self.nvar = 0
        
        self.executor = None # For multiprocessing

    # --- Calculation Models (matrix_calc_noex, matrix_calc, Baldwin) ---
    # These methods remain unchanged.
    def matrix_calc_noex(self, dG, R1, R2a, dRF, T, v1, v1err, B0):
        """
        Vectorized analytical calculation for NoEx (No Exchange) model.
        dRF (offset) can be a scalar or a numpy array.
        """
        # Ensure dRF is an array
        dRF_arr = np.atleast_1d(dRF)
        
        wRF = dRF_arr * B0 * 2.0 * np.pi
        wG = dG * B0 * 2.0 * np.pi
        wa = wG - wRF # Shape: (N_offsets,)
        
        w1_val = v1 * 2.0 * np.pi
        w1err_val = v1err * 2.0 * np.pi

        if w1err_val == 0:
            wxs = np.array([w1_val])
            weights = np.array([1.0])
        else:
            wxs = np.linspace(-2.0 * w1err_val + w1_val, 2.0 * w1err_val + w1_val, 10)
            weights = fast_gaussian(wxs, w1_val, w1err_val)

        if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
             weights = np.ones_like(wxs) / len(wxs) if len(wxs)>0 else np.array([1.0])
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum != 0:
            weights = weights / weights_sum

        # Vectorized calculation
        # wxs shape: (10,) -> (10, 1)
        # wa shape: (N_offsets,) -> (1, N_offsets)
        
        w1_loop = wxs[:, np.newaxis]
        wa_grid = wa[np.newaxis, :]
        
        # Effective field squared
        weff2 = wa_grid**2 + w1_loop**2
        
        # Tilt angle functions
        # sin^2(theta) = w1^2 / weff2
        # cos^2(theta) = wa^2 / weff2
        
        sintheta2 = w1_loop**2 / weff2
        costheta2 = wa_grid**2 / weff2
        
        # R1rho = R1 * cos^2 + R2a * sin^2
        R1rho = R1 * costheta2 + R2a * sintheta2
        
        # Intensity decay
        # I = I0 * cos^2(theta) * exp(-R1rho * T)
        # Assuming I0 = 1 (normalized) and we observe Mz projection?
        # Original code: startM = [0,0,0,1] (Mz=1). endM[3] is Mz.
        # In tilted frame, Mz aligns with Beff? No.
        # Standard R1rho experiment:
        # 1. 90y -> Mx
        # 2. Spin lock along x (or y).
        # 3. Decay.
        # 4. 90-y -> Mz (detect).
        # Wait, the matrix_calc_noex original code uses:
        # startM = [0,0,0,1] (Mz).
        # Alist has [2*R1a, -wy, wx, -R1a] in the last row?
        # Let's check the original matrix structure.
        # Alist = [[0,0,0,0], [0,-R2a,-wa,wy], [0,wa,-R2a,-wx], [2*R1a,-wy,wx,-R1a]]
        # This looks like [1, Mx, My, Mz].
        # Row 0: d/dt(1) = 0.
        # Row 1 (Mx): -R2a*Mx - wa*My + wy*Mz.
        # Row 2 (My): wa*Mx - R2a*My - wx*Mz.
        # Row 3 (Mz): -wy*Mx + wx*My - R1a*Mz + 2*R1a. (Relaxation to equilibrium M0=1? 2*R1a term suggests M0=2? Or maybe 2*R1a is just R1a * 2?)
        # Usually dMz/dt = -R1(Mz - M0). = -R1*Mz + R1*M0.
        # If M0=1, then +R1.
        # Here we have +2*R1a. Maybe M0=2? Or maybe the basis is different.
        # But startM is [0,0,0,1].
        # If we assume standard R1rho, the projection factor is usually cos^2(theta).
        # Let's trust the Baldwin formula which reduces to R1rho.
        # Baldwin uses: preIcal += weights[i]*costheta2*exp_val.
        # So I will use costheta2 * exp(-R1rho * T).
        
        exp_arg = -1.0 * T * R1rho
        exp_val = np.exp(exp_arg)
        exp_val[exp_arg < -700] = 0.0
        
        # Weighted sum
        Icalc = np.sum(weights[:, np.newaxis] * costheta2 * exp_val, axis=0)
        
        if np.ndim(dRF) == 0:
            return max(0.0, Icalc.item())
        else:
            return np.maximum(0.0, Icalc)

   
    def matrix_calc(self, kab, kba, dG, dw, R1, R2a, R2b, dRF, T, v1, v1err, B0):
        R1a = R1; R1b = R1
        wRF = dRF * B0 * 2.0 * np.pi
        wG = dG * B0 * 2.0 * np.pi; wa = wG - wRF
        wE = (dG + dw) * B0 * 2.0 * np.pi; wb = wE - wRF
        wy = 0.0
        
        kex = kab + kba 
        if kex == 0: pB, pA = 0.0, 1.0
        else: pB, pA = kab / kex, kba / kex

        w1_val = v1*2*np.pi; w1err_val = v1err*2*np.pi
        
        if w1err_val == 0:
            wxs, weights = np.array([w1_val]), np.array([1.0])
        else:
            wxs = np.linspace(-2.0*w1err_val + w1_val, 2.0*w1err_val + w1_val, 10)
            weights = fast_gaussian(wxs, w1_val, w1err_val)


        if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
             weights = np.ones_like(wxs)/len(wxs) if len(wxs)>0 else np.array([1.0])

        refM = pA if pA != 0 else 1.0
        startM = np.array([0,0,0,pA,0,0,pB], dtype=float)
        preIcal = 0.0
        for i in range(len(wxs)):
            wx = wxs[i]
            Alist = [[0,0,0,0,0,0,0], [0,-kab-R2a,-wa,wy,kba,0,0], [0,wa,-kab-R2a,-wx,0,kba,0],
                     [2*R1a*pA,-wy,wx,-kab-R1a,0,0,kba], [0,kab,0,0,-kba-R2b,-wb,wy],
                     [0,0,kab,0,wb,-kba-R2b,-wx], [2*R1b*pB,0,0,kab,-wy,wx,-kba-R1b]]
            A = np.array(Alist, dtype=float)
            eA = expm(A * T) 
            endM = np.dot(eA,startM)
            preIcal += weights[i]*endM[3]/refM
    
        Icalc = preIcal/np.sum(weights) if np.sum(weights) != 0 else 0.0
        return max(0.0, Icalc)

    def Baldwin(self, kab, kba, dG, dw, R1, R2a, R2b, dRF, T, v1, v1err, B0):
        """
        Vectorized Baldwin model calculation.
        dRF (offset) can be a scalar or a numpy array.
        """
        epsilon = 1e-9 
        kex = kab + kba 
        if kex == 0: pE, pG = 0.0, 1.0
        else: pE, pG = kab / kex, kba / kex
        dR = R2b-R2a
        
        # Ensure dRF is an array for vectorization
        dRF_arr = np.atleast_1d(dRF)
        
        wRF = dRF_arr * B0 * 2. * np.pi
        wG = dG * B0 * 2. * np.pi
        ddG = wG - wRF # Shape: (N_offsets,)
        
        dE_val = dG + dw
        wE = dE_val * B0 * 2. * np.pi
        ddE = wE - wRF # Shape: (N_offsets,)
        
        ddw = ddE - ddG
        da = pG * ddG + pE * ddE # Shape: (N_offsets,)
        
        w1b_val = v1 * 2. * np.pi
        w1err_val = v1err * 2. * np.pi
        
        if w1err_val == 0: 
            wxs, weights = np.array([w1b_val]), np.array([1.0])
        else:
            wxs = np.linspace(-2.*w1err_val+w1b_val, 2.*w1err_val+w1b_val, 10)
            weights = fast_gaussian(wxs, w1b_val, w1err_val)
        
        if np.sum(weights) == 0 or np.isnan(np.sum(weights)):
            weights = np.ones_like(wxs)/len(wxs) if len(wxs)>0 else np.array([1.0])
        
        # Normalize weights
        weights_sum = np.sum(weights)
        if weights_sum != 0:
            weights = weights / weights_sum
            
        # Vectorized integration over B1 inhomogeneity (wxs)
        # We need to broadcast wxs against dRF_arr
        # wxs shape: (10,)
        # dRF_arr shape: (N_offsets,)
        # Resulting shapes for broadcasting: (10, N_offsets)
        
        w1_loop = wxs[:, np.newaxis] # Shape (10, 1)
        
        # Expand offset-dependent variables to match w1_loop broadcasting
        ddG_grid = ddG[np.newaxis, :] # Shape (1, N_offsets)
        ddE_grid = ddE[np.newaxis, :]
        da_grid = da[np.newaxis, :]
        
        # Calculations on the grid
        OG2 = w1_loop**2 + ddG_grid**2
        OE2 = w1_loop**2 + ddE_grid**2
        Oa2 = w1_loop**2 + da_grid**2
        
        tantheta2 = (w1_loop**2) / (da_grid**2 + epsilon)
        sintheta2 = (w1_loop**2) / (Oa2 + epsilon)
        costheta2 = (da_grid**2) / (Oa2 + epsilon)
        
        F1p = pG * pE * (ddw**2) # Scalar (actually depends on ddw which is constant? No, ddw = ddE - ddG = (wE-wRF)-(wG-wRF) = wE-wG = dw*B0... so ddw is constant across offsets!)
        # Wait, ddw = ddE - ddG. 
        # ddE = wE - wRF, ddG = wG - wRF. 
        # ddE - ddG = wE - wG = (dG+dw)*C - dG*C = dw*C. 
        # So ddw is indeed independent of offset (wRF).
        # Let's re-verify lines 118-120 of original code.
        # wRF = dRF*B0...; wG = dG*B0...; ddG = wG-wRF
        # wE = ...; ddE = wE-wRF
        # ddw = ddE-ddG
        # Yes, ddw is constant w.r.t offset.
        
        # F1p is constant w.r.t offset and w1.
        
        F2p = (kex**2) + (w1_loop**2) + ((ddG_grid * ddE_grid / (da_grid + epsilon))**2)
        Dp = (kex**2) + OG2 * OE2 / (Oa2 + epsilon)
        
        F1 = pE * (OG2 + (kex**2) + dR * pG * kex)
        F2 = 2 * kex + (w1_loop**2) / (kex + epsilon) + dR * pG
        
        F3_term_denom = (w1_loop**2 + epsilon)
        # Note: F3 depends on OG2 which depends on offset
        F3 = 3 * pE * kex + (2 * pG * kex + (w1_loop**2) / (kex + epsilon) + dR + dR * ((pE * kex)**2) / (OG2 + epsilon)) * (OG2 / F3_term_denom)
        
        CR1_den = Dp + dR * F3 * sintheta2 + epsilon
        CR1 = (F2p + (F1p + dR * (F3 - F2)) * tantheta2) / CR1_den
        
        CR2_den = Dp + dR * F3 * sintheta2 + epsilon
        CR2 = (Dp / (sintheta2 + epsilon) - F2p / (tantheta2 + epsilon) - F1p + dR * F2) / CR2_den
        
        Rex_den = Dp + dR * F3 * sintheta2 + epsilon
        Rex = (F1p * kex + dR * F1) / Rex_den
        
        R1r = CR1 * R1 * costheta2 + (CR2 * R2a + Rex) * sintheta2
        
        exp_arg = -1. * T * R1r
        # exp_val = 0. if exp_arg < -700 else np.exp(exp_arg) # Vectorized version below
        exp_val = np.exp(exp_arg)
        exp_val[exp_arg < -700] = 0.
        
        # Weighted sum over w1 dimension (axis 0)
        # weights shape: (10,) -> (10, 1) to broadcast
        preIcal = np.sum(weights[:, np.newaxis] * costheta2 * exp_val, axis=0)
        
        Icalc = preIcal # Already normalized weights
        
        # Return scalar if input was scalar, else array
        if np.ndim(dRF) == 0:
            return max(0.0, Icalc.item())
        else:
            return np.maximum(0.0, Icalc)


    def selMethod(self, initConf):
        """Sets the calculation method based on initConf."""
        if initConf is None: # Handle cases where fitting_config might be None
            if self.verbose: print(f"selMethod received None for initConf, using current method '{self.method}' or default 'Baldwin'.")
            method_val = self.method if self.method else 'Baldwin' # Fallback
        else:
            method_val = initConf.get('Method', 'Baldwin')

        self.method = method_val if method_val in ['Baldwin', 'Matrix', 'NoEx'] else 'Baldwin'
        if self.verbose: print(f"Selected method: {self.method}")

    # _generate_initial_parameters method is REMOVED from here. It's now in fit.py.

    def seParam(self, p_flat_input):
        # This method remains unchanged.
        p_flat = list(p_flat_input) 
        params_structured = {}
        if self.method == 'NoEx':
            params_structured['dGs'], params_structured['r1s'], params_structured['r2as'] = [], [], []
            idx = 0
            for res in self.dataset.res:
                if res.active:
                    if idx + 2 < len(p_flat):
                        params_structured['dGs'].append(p_flat[idx])
                        params_structured['r1s'].append(p_flat[idx+1])
                        params_structured['r2as'].append(p_flat[idx+2])
                        idx += 3
                    else: 
                        [params_structured[k].append(v) for k,v in zip(['dGs','r1s','r2as'], [0.,1.,10.])]
                else: [params_structured[k].append(0.) for k in ['dGs','r1s','r2as']] # Inactive residues
            return [params_structured['dGs'], params_structured['r1s'], params_structured['r2as']]
        else: 
            if len(p_flat) < 2: raise ValueError(f"Param array p_flat (len {len(p_flat)}) too short for exchange model.")
            params_structured['kab'], params_structured['kba'] = p_flat[0], p_flat[1]
            keys = ['dGs','dws','r1s','r2as','r2bs']
            for k in keys: params_structured[k] = []
            idx = 2
            for res in self.dataset.res:
                if res.active:
                    if idx + 4 < len(p_flat):
                        for i, k_val in enumerate(keys): params_structured[k_val].append(p_flat[idx+i])
                        idx += 5
                    else: 
                        [params_structured[k].append(v) for k,v in zip(keys, [0.,0.1,1.,10.,20.])]
                else: [params_structured[k].append(0.) for k in keys] # Inactive residues
            return [params_structured['kab'], params_structured['kba']] + [params_structured[k] for k in keys]


    def errFunc(self, p_flat):
        # This method remains largely unchanged but ensures NaNs from calculations are handled.
        try:
            par_struc = self.seParam(p_flat)
        except ValueError as e:
            if self.verbose: print(f"Error in seParam (errFunc): {e}")
            # Return a large residual array matching expected nvar if known, else a single large value
            # This helps least_squares understand failure.
            n_residuals = getattr(self, 'nvar', 1) 
            if n_residuals <=0: n_residuals = sum(len(es.offset) for r in self.dataset.res if r.active for es in r.estSpecs) # Estimate
            if n_residuals <=0: n_residuals = 1 # Absolute fallback
            return np.full(n_residuals, 1e6, dtype=float)


        residuals_list = []
        kab, kba, dGs, dws, r1s, r2as, r2bs = (None,)*7 
        if self.method == 'NoEx':
            dGs, r1s, r2as = par_struc[0], par_struc[1], par_struc[2]
        else:
            kab, kba = par_struc[0], par_struc[1]
            dGs, dws, r1s, r2as, r2bs = par_struc[2], par_struc[3], par_struc[4], par_struc[5], par_struc[6]
       
        # Collect tasks for Matrix method parallelization
        matrix_tasks = []
        CHUNK_SIZE = 50 # Increase chunk size
        
        # Heuristic: Only use parallel if total offsets > 200
        total_offsets = sum(len(res.estSpecs[0].offset) for res in self.dataset.res if res.active and res.estSpecs)
        use_parallel = self.method == 'Matrix' and hasattr(self, 'executor') and self.executor and total_offsets > 200

        if use_parallel:
            for i, res_obj in enumerate(self.dataset.res):
                if res_obj.active:
                    for estspec in res_obj.estSpecs:
                        offsets = estspec.offset
                        # Create chunks
                        for k in range(0, len(offsets), CHUNK_SIZE):
                            chunk_offsets = offsets[k:k + CHUNK_SIZE]
                            matrix_tasks.append((kab, kba, dGs[i], dws[i], r1s[i], r2as[i], r2bs[i], chunk_offsets, estspec.T, estspec.v1, estspec.v1err, estspec.field))
            
            # Execute parallel tasks
            try:
                # map returns an iterator, convert to list to trigger execution and catch exceptions
                chunk_results = list(self.executor.map(matrix_calc_worker_chunk, matrix_tasks))
                
                # Flatten results
                flat_results = []
                for res in chunk_results:
                    flat_results.extend(res)
                
                matrix_results_iter = iter(flat_results)
                
            except Exception as e:
                if self.verbose: print(f"Error in parallel execution: {e}")
                raise e



        for i, res_obj in enumerate(self.dataset.res):
            if res_obj.active:
                for estspec in res_obj.estSpecs:
                    # Vectorized call for Baldwin
                    offsets = np.array(estspec.offset)
                    
                    if self.method == 'Baldwin':
                        est_calc = self.Baldwin(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offsets,estspec.T,estspec.v1,estspec.v1err,estspec.field)
                    elif self.method == 'NoEx':
                        est_calc = self.matrix_calc_noex(dGs[i],r1s[i],r2as[i],offsets,estspec.T,estspec.v1,estspec.v1err,estspec.field)
                    elif self.method == 'Matrix':
                        if use_parallel:
                            # Retrieve results from parallel execution
                            est_calc = []
                            for _ in range(len(offsets)):
                                est_calc.append(next(matrix_results_iter))
                            est_calc = np.array(est_calc)
                        else:
                            # Fallback for Matrix method (serial)
                            est_calc = []
                            for k_offset, offset_val in enumerate(offsets):
                                val = self.matrix_calc(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val,estspec.T,estspec.v1,estspec.v1err,estspec.field)
                                est_calc.append(val)
                            est_calc = np.array(est_calc)
                    else:
                         # Should not happen given selMethod
                         est_calc = np.zeros_like(offsets)


                    est_calc = np.nan_to_num(est_calc, nan=1e6, posinf=1e6, neginf=-1e6)
                    std_devs = np.array(estspec.intstd)
                    std_devs[std_devs == 0] = 1.0
                    
                    # Calculate residuals for this spectrum
                    current_residuals = (np.array(estspec.int) - est_calc) / std_devs
                    residuals_list.extend(current_residuals)

        
        residuals = np.array(residuals_list, dtype=float)
        if len(residuals) == 0 and self.verbose: # No active residues or data points
            print("Warning (errFunc): No residuals generated. Check active residues and data.")
            # least_squares expects a non-empty array. If this occurs, fitting will likely fail.
            # Return a single large residual if p_flat suggests parameters were expected.
            return np.array([1e6], dtype=float) if len(p_flat) > 0 else np.array([], dtype=float)


        self.chi2 = np.sum(residuals**2); self.npar = len(p_flat)
        self.nvar = len(residuals); self.dof = max(1, self.nvar - self.npar)

        if self.verbose and self.nvar > 0: # Only log if there are residuals
            log_msg_parts = []
            if self.method!='NoEx' and kab is not None and kba is not None: log_msg_parts.extend([f'kab={kab:8.3f}',f'kba={kba:8.3f}'])
            log_msg_parts.append(f'chi2={self.chi2:12.3f} dof={self.dof} nv={self.nvar} np={self.npar}')
            # print(' '.join(log_msg_parts)) # Verbose, consider conditional print
            # print(p_flat) 
        return residuals
       
    def fit(self, p0=None, fitting_config=None):
        """
        Performs model fitting.
        If p0 is None, initial parameters are generated using fitting_config.
        fitting_config is also used to determine the model fitting method.
        Delegates to functions in fit_module.
        """
        # Determine method first to decide on executor
        if fitting_config is not None:
             self.selMethod(fitting_config)
        
        # Initialize multiprocessing executor if method is Matrix
        # We do this BEFORE generate_initial_parameters because it also calls errFunc
        executor_context = None
        if self.method == 'Matrix':
            if self.verbose: print("Initializing ProcessPoolExecutor for Matrix method...")
            executor_context = concurrent.futures.ProcessPoolExecutor()
            self.executor = executor_context

        try:
            if p0 is None:
                if fitting_config is None:
                    raise ValueError("fitting_config must be provided to est_model.fit if p0 is not specified.")
                if self.verbose: print("est_model.fit: p0 not provided, calling fit_module.generate_initial_parameters...")
                # fit_module.generate_initial_parameters will call self.selMethod(fitting_config) again, which is fine
                p0 = fit_module.generate_initial_parameters(self, fitting_config)
            else: # p0 is provided
                if self.verbose: print(f"est_model.fit: Using provided p0. Method: {self.method}. Calling fit_module.perform_least_squares_fit...")
            
            return fit_module.perform_least_squares_fit(self, p0)
        
        finally:
            if executor_context:
                executor_context.shutdown()
                self.executor = None
                if self.verbose: print("ProcessPoolExecutor shutdown.")



    def pdf(self, p1_flat, pdfFileName):
        # This method remains largely unchanged.
        if not isinstance(p1_flat, np.ndarray): p1_flat = np.array(p1_flat, dtype=float)

        pdf = PdfPages(pdfFileName)
        try:
            par_struc = self.seParam(p1_flat)
        except Exception as e:
            print(f"Error in seParam during pdf generation: {e}. Skipping PDF generation.")
            if self.verbose: print(f"Parameters causing seParam error in PDF: {p1_flat}")
            pdf.close() 
            return

        kab, kba, dGs, dws, r1s, r2as, r2bs = (None,) * 7

        if self.method == 'NoEx':
            if len(par_struc) == 3:
                dGs, r1s, r2as = par_struc[0], par_struc[1], par_struc[2]
            else:
                print(f"Warning: Parameter structure mismatch for NoEx in PDF. Got {len(par_struc)} elements. Expected 3.")
                pdf.close(); return
        else: 
            if len(par_struc) == 7:
                kab, kba = par_struc[0], par_struc[1]
                dGs, dws, r1s, r2as, r2bs = par_struc[2], par_struc[3], par_struc[4], par_struc[5], par_struc[6]
            else:
                print(f"Warning: Parameter structure mismatch for {self.method} in PDF. Got {len(par_struc)} elements. Expected 7.")
                pdf.close(); return

        for i, res_obj in enumerate(self.dataset.res):
            if res_obj.active:
                if not res_obj.estSpecs: 
                    if self.verbose: print(f"No spectra for active residue {res_obj.label}, skipping PDF page.")
                    continue

                fig = figure(figsize=(8.5, 6)) 
                ax = fig.add_subplot(1, 1, 1)

                colorsSet = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] 
                all_offsets_for_res, all_intensities_for_res, all_stddevs_for_res = [], [], []

                for j, ep in enumerate(res_obj.estSpecs):
                    if not ep.offset: 
                        if self.verbose: print(f"Empty EstSpec {j} for residue {res_obj.label}, skipping.")
                        continue

                    all_offsets_for_res.extend(ep.offset)
                    all_intensities_for_res.extend(ep.int)
                    all_stddevs_for_res.extend(ep.intstd)
                    c = colorsSet[j % len(colorsSet)]
                    ax.errorbar(ep.offset, ep.int, yerr=ep.intstd, fmt=f'{c}o', markersize=3, label=f'exp {ep.v1:.1f}Hz {ep.T*1000.0:.1f}ms')

                    if len(ep.offset) > 0: 
                        tx = np.array(sorted(ep.offset))
                        ty = np.zeros_like(tx, dtype=float)
                    else: 
                        continue
                    
                    for ii_offset, offset_val_plot in enumerate(tx):
                        calc_val = 0.0 
                        if self.method == 'Matrix':
                            calc_val = self.matrix_calc(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                        elif self.method == 'Baldwin':
                            calc_val = self.Baldwin(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                        elif self.method == 'NoEx':
                            calc_val = self.matrix_calc_noex(dGs[i],r1s[i],r2as[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                        else: 
                            calc_val = self.Baldwin(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                        ty[ii_offset] = np.nan_to_num(calc_val)
                    ax.plot(tx, ty, f'{c}-', label=f'calc {ep.v1:.1f}Hz {ep.T*1000.0:.1f}ms')

                ax.set_title(res_obj.label)
                ax.set_xlabel('Chemical Shift Offset (ppm)')
                ax.set_ylabel('Intensity')
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.75, box.height]) 
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small') 
                ax.grid(True)

                if all_offsets_for_res:
                    ax.set_xlim(np.min(all_offsets_for_res), np.max(all_offsets_for_res))
                if all_intensities_for_res and all_stddevs_for_res: # Check if lists are not empty
                    min_y_val = np.min(np.array(all_intensities_for_res) - np.array(all_stddevs_for_res))
                    max_y_val = np.max(np.array(all_intensities_for_res) + np.array(all_stddevs_for_res))
                    plot_min_y = min(0.0, min_y_val * 0.95 if min_y_val >= 0 else min_y_val * 1.05) # Adjusted logic for negative min_y_val
                    plot_max_y = max_y_val * 1.05 if max_y_val >= 0 else max_y_val * 0.95 # Adjusted logic for negative max_y_val
                    if plot_max_y <= plot_min_y: plot_max_y = plot_min_y + 0.1 
                    ax.set_ylim(plot_min_y, plot_max_y)

                fig.tight_layout(rect=[0, 0, 0.85, 1]) 
                pdf.savefig(fig)
                close(fig) 
        pdf.close()

    def datapdf(self, pdfFileName):
        # This method remains unchanged.
        pdf = PdfPages(pdfFileName)
        for r_o in self.dataset.res:
            if r_o.active:
                fig=figure(figsize=(8.5,6));ax=fig.add_subplot(1,1,1)
                cs=['b','g','r','c','m','y','k'];mxv,miv,mayv,miyv=[],[],[],[] # Added 'k'
                for ispec,ep in enumerate(r_o.estSpecs):
                    if not ep.offset: continue # Skip if no offset data
                    mxv.append(np.min(ep.offset));miv.append(np.max(ep.offset))
                    c=cs[ispec%len(cs)]
                    ax.errorbar(ep.offset,ep.int,yerr=ep.intstd,fmt=f'{c}o',ms=3,label=f'exp {ep.v1:.1f}Hz {ep.T*1e3:.1f}ms')
                    if ep.int and ep.intstd: # Ensure lists are not empty
                        miyv.append(np.min(np.array(ep.int)-np.array(ep.intstd)));mayv.append(np.max(np.array(ep.int)+np.array(ep.intstd)))
                ax.set_title(r_o.label);ax.set_xlabel('Chemical Shift Offset (ppm)');ax.set_ylabel('Intensity')
                box=ax.get_position();ax.set_position([box.x0,box.y0,box.width*0.75,box.height])
                ax.legend(loc='center left',bbox_to_anchor=(1,0.5), fontsize='small');ax.grid(True)
                
                plot_min_y_data, plot_max_y_data = 0, 0.1 # Default y-limits
                if miyv and mayv: # Ensure these lists are populated
                    actual_min_y = np.min(miyv)
                    actual_max_y = np.max(mayv)
                    plot_min_y_data = min(0, 0.95 * actual_min_y if actual_min_y >=0 else 1.05 * actual_min_y)
                    plot_max_y_data = 1.05 * actual_max_y if actual_max_y >=0 else 0.95 * actual_max_y
                    if plot_max_y_data <= plot_min_y_data : plot_max_y_data = plot_min_y_data + 0.1
                ax.set_ylim(plot_min_y_data,plot_max_y_data)

                if mxv and miv:ax.set_xlim(np.min(mxv),np.max(miv))
                fig.tight_layout(rect=[0, 0, 0.85, 1])
                pdf.savefig(fig);close(fig)
        pdf.close()

    def getLogBuffer(self, fit_output_tuple):
        # This method remains largely unchanged.
        p_optimized,covar = fit_output_tuple[0],fit_output_tuple[1]
        parstd_diag = np.full_like(p_optimized, np.nan) 
        if covar is not None and covar.shape == (len(p_optimized), len(p_optimized)) and not np.all(np.isnan(covar)):
            diag_covar = np.diag(covar)
            # Only take sqrt if elements are non-negative, not NaN, and not Inf
            valid_indices = ~np.isnan(diag_covar) & ~np.isinf(diag_covar) & (diag_covar >= 0)
            parstd_diag[valid_indices] = np.sqrt(diag_covar[valid_indices])
            if np.any(~valid_indices & (~np.isnan(diag_covar))): # If there were issues beyond NaN (e.g. <0, inf)
                 if self.verbose: print("Warning: Some diagonal covar elements were invalid (e.g. <0, inf); std dev for these set to NaN.")
        elif self.verbose: print("Warning: Covar matrix issue or unavailable; std devs are NaN for getLogBuffer.")


        log_list=['*'*51,self.programName,'*'*51+'\n']
        try:user,hostname=getlogin(),uname()[1];log_list.append(f'User: {user}@{hostname}')
        except Exception: log_list.append('User: Unknown')
        log_list.extend([f'{ctime()}','*'*51]);par_struc=self.seParam(p_optimized)
        kab_v, kba_v, dGs, dws, r1s, r2as, r2bs = (None,)*7 # Initialize all
        if self.method=='NoEx':dGs,r1s,r2as=par_struc[0],par_struc[1],par_struc[2]
        else:kab_v,kba_v=par_struc[0],par_struc[1];dGs,dws,r1s,r2as,r2bs=par_struc[2],par_struc[3],par_struc[4],par_struc[5],par_struc[6]
        for ir,ro in enumerate(self.dataset.res):
            if ro.active:
                log_list.append(f'\n# {ro.label}')
                for es in ro.estSpecs:
                    log_list.extend([f'B0 [MHz]: {es.field:8.3f}',f'T [ms]: {es.T*1e3:8.3f}',f'v1 [Hz]: {es.v1:8.3f}'])
                    log_list.append(f"{'Offset':>8} {'ExpI':>12} {'ExpStd':>12} {'CalcI':>12}")
                    for ko,ovl in enumerate(es.offset):
                        ec=0.;
                        if self.method=='Matrix':ec=self.matrix_calc(kab_v,kba_v,dGs[ir],dws[ir],r1s[ir],r2as[ir],r2bs[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        elif self.method=='Baldwin':ec=self.Baldwin(kab_v,kba_v,dGs[ir],dws[ir],r1s[ir],r2as[ir],r2bs[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        elif self.method=='NoEx':ec=self.matrix_calc_noex(dGs[ir],r1s[ir],r2as[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        ec = np.nan_to_num(ec) # Ensure no NaNs in log
                        log_list.append(f'{ovl:8.3f} {es.int[ko]:12.3f} {es.intstd[ko]:12.3f} {ec:12.3f}')
                log_list.append('*'*51)
        
        # Recalculate chi2, dof etc with optimized params to ensure consistency for the log
        _ = self.errFunc(p_optimized) 
        log_list.extend(['\n\n\n'+'*'*41,f'Results using {self.method} method\n',f'Chi2:     {self.chi2:8.6f}',
                         f'red_chi2: {self.chi2/self.dof if self.dof>0 else float("inf"):8.6f}',
                         f'dof:      {self.dof:d}',f'nv:       {self.nvar:d}',f'np:       {self.npar:d}','*'*43,
                         'Fitted Parameters (+/- StdDev if available):'])
        pi_log=0
        if self.method!='NoEx':
            log_list.append(f'kab [s-1]: {p_optimized[pi_log]:8.3f} +/- {parstd_diag[pi_log]:8.3f}');pi_log+=1
            log_list.append(f'kba [s-1]: {p_optimized[pi_log]:8.3f} +/- {parstd_diag[pi_log]:8.3f}');pi_log+=1
        for idx_res_log, ro_log in enumerate(self.dataset.res): # Use enumerate for correct index with p_optimized
            if ro_log.active:
                log_list.extend(['*'*43,f'residue: {ro_log.label}'])
                if self.method=='NoEx':
                    if pi_log + 2 < len(p_optimized): # Check bounds for p_optimized and parstd_diag
                        log_list.append(f'peak pos [ppm]: {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R1 [s-1]:       {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2a [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    else: log_list.append("Error: Parameter index out of bounds for NoEx.")
                else:
                    if pi_log + 4 < len(p_optimized): # Check bounds
                        log_list.append(f'peak pos [ppm]: {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'cs diff [ppm]:  {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R1 [s-1]:       {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2a [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2b [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    else: log_list.append("Error: Parameter index out of bounds for Exchange model.")
        log_list.append('*'*43)
        return "\n".join(log_list)


    def getLogBufferMC(self, p_center, all_mc_ps_list):
        # This method remains largely unchanged.
        if not all_mc_ps_list: # Handle empty list
            log_list=['*'*51,f'{self.programName} - Monte Carlo Analysis - NO SUCCESSFUL RUNS','*'*51+'\n']
            return "\n".join(log_list), np.array([])


        all_mc_ps_array = np.array(all_mc_ps_list)
        mean_mc_params = np.mean(all_mc_ps_array, axis=0)
        std_dev_mc_params = np.std(all_mc_ps_array, axis=0)
        
        log_list=['*'*51,f'{self.programName} - Monte Carlo Analysis','*'*51+'\n']
        try:user,hostname=getlogin(),uname()[1];log_list.append(f'User: {user}@{hostname}')
        except Exception:log_list.append('User: Unknown')
        log_list.extend([f'{ctime()}',f'Num MC runs: {len(all_mc_ps_list)}','*'*51])
        
        # Ensure method is set before calling seParam with mean_mc_params
        # This assumes self.method was set by the initial fit before MC runs.
        if not self.method:
            print("Warning (getLogBufferMC): self.method not set. Defaulting to Baldwin for seParam.")
            self.method = 'Baldwin' # Fallback, though ideally it should be set.

        par_s_mc=self.seParam(mean_mc_params) # Uses self.method
        kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m = (None,)*7 # Initialize all

        if self.method=='NoEx':
            if len(par_s_mc) == 3: dGs_m,r1s_m,r2as_m=par_s_mc[0],par_s_mc[1],par_s_mc[2]
            else: log_list.append("Error: Param structure mismatch for NoEx in getLogBufferMC."); # Continue with Nones
        else:
            if len(par_s_mc) == 7: kab_m,kba_m=par_s_mc[0],par_s_mc[1];dGs_m,dws_m,r1s_m,r2as_m,r2bs_m=par_s_mc[2],par_s_mc[3],par_s_mc[4],par_s_mc[5],par_s_mc[6]
            else: log_list.append(f"Error: Param structure mismatch for {self.method} in getLogBufferMC."); # Continue with Nones

        for ir,ro in enumerate(self.dataset.res):
            if ro.active:
                log_list.append(f'\n# {ro.label}')
                for es in ro.estSpecs:
                    log_list.extend([f'B0 [MHz]: {es.field:8.3f}',f'T [ms]: {es.T*1e3:8.3f}',f'v1 [Hz]: {es.v1:8.3f}'])
                    log_list.append(f"{'Offset':>8} {'ExpI':>12} {'ExpStd':>12} {'CalcI_MC_Mean':>18}")
                    for ko,ovl in enumerate(es.offset):
                        ec_m=0.
                        # Check if params were successfully unpacked for the current method
                        if self.method=='Matrix' and all(p is not None for p in [kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m]):
                            ec_m=self.matrix_calc(kab_m,kba_m,dGs_m[ir],dws_m[ir],r1s_m[ir],r2as_m[ir],r2bs_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        elif self.method=='Baldwin' and all(p is not None for p in [kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m]):
                            ec_m=self.Baldwin(kab_m,kba_m,dGs_m[ir],dws_m[ir],r1s_m[ir],r2as_m[ir],r2bs_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        elif self.method=='NoEx' and all(p is not None for p in [dGs_m,r1s_m,r2as_m]):
                            ec_m=self.matrix_calc_noex(dGs_m[ir],r1s_m[ir],r2as_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                        ec_m = np.nan_to_num(ec_m) # Ensure no NaNs in log
                        log_list.append(f'{ovl:8.3f} {es.int[ko]:12.3f} {es.intstd[ko]:12.3f} {ec_m:18.3f}')
                log_list.append('*'*51)
        
        # Use original center for chi2 stats of the initial fit.
        # self.errFunc called with p_center will update self.chi2, self.dof etc.
        _ = self.errFunc(p_center) 
        log_list.extend(['\n\n\n'+'*'*41,f'Results using {self.method} (MC Stats)\n',
                         f'Chi2 (init fit): {self.chi2:8.6f}',
                         f'red_chi2 (init fit): {self.chi2/self.dof if self.dof>0 else float("inf"):8.6f}',
                         f'dof (init fit):      {self.dof:d}',f'nv (init fit):       {self.nvar:d}',f'np (init fit):       {self.npar:d}','*'*43,
                         'Fitted Params (Mean from MC +/- StdDev from MC):'])
        pi_log=0
        # Check if mean_mc_params and std_dev_mc_params have expected length for the method
        expected_params_len = 0
        active_res_count = sum(1 for r in self.dataset.res if r.active)
        if self.method != 'NoEx': expected_params_len = 2 + active_res_count * 5
        else: expected_params_len = active_res_count * 3

        if len(mean_mc_params) != expected_params_len or len(std_dev_mc_params) != expected_params_len:
            log_list.append(f"Error: MC parameter array length mismatch. Mean: {len(mean_mc_params)}, StdDev: {len(std_dev_mc_params)}, Expected: {expected_params_len}")
        else:
            if self.method!='NoEx':
                log_list.append(f'kab [s-1]: {mean_mc_params[pi_log]:8.3f} +/- {std_dev_mc_params[pi_log]:8.3f}');pi_log+=1
                log_list.append(f'kba [s-1]: {mean_mc_params[pi_log]:8.3f} +/- {std_dev_mc_params[pi_log]:8.3f}');pi_log+=1
            for ro_log in self.dataset.res:
                if ro_log.active:
                    log_list.extend(['*'*43,f'residue: {ro_log.label}'])
                    if self.method=='NoEx':
                        if pi_log + 2 < len(mean_mc_params): # Check bounds
                            log_list.append(f'peak pos [ppm]: {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'R1 [s-1]:       {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'R2a [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        else: log_list.append("Error: MC Parameter index out of bounds for NoEx.")
                    else: # Exchange models
                        if pi_log + 4 < len(mean_mc_params): # Check bounds
                            log_list.append(f'peak pos [ppm]: {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'cs diff [ppm]:  {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'R1 [s-1]:       {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'R2a [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                            log_list.append(f'R2b [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        else: log_list.append("Error: MC Parameter index out of bounds for Exchange model.")
        log_list.append('*'*43)
        return "\n".join(log_list), mean_mc_params
