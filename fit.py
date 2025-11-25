#!/usr/bin/env python
# fit.py
# Contains fitting logic extracted from estmodel.py
# Original estmodel.py by Donghan Lee 2015
# Refactoring and scipy.optimize.least_squares integration: 2025

import numpy as np
from scipy.optimize import least_squares

def generate_initial_parameters(model_instance, initConf):
    """
    Generates the initial parameter list p0 based on initConf.
    This function is similar to the original _generate_initial_parameters
    from estmodel.py.
    It requires model_instance to access dataset, selMethod, verbose, and errFunc (for chi2 in grid search).
    """
    model_instance.selMethod(initConf) # Set model_instance.method based on initConf

    p_initial_list = []
    if model_instance.method == 'NoEx':
        for res_obj in model_instance.dataset.res:
            if res_obj.active:
                if not res_obj.estSpecs or not res_obj.estSpecs[0].int:
                    if model_instance.verbose: print(f"Warning (fit.py): No data for {res_obj.label}, using defaults for NoEx.")
                    p_initial_list.extend([0.0, 1.0, 10.0])
                    continue
                first_spec = res_obj.estSpecs[0]
                dG = 0.0
                if len(first_spec.offset) > 0 and len(first_spec.int) > 0:
                    try:
                        dG = first_spec.offset[np.argmin(first_spec.int)]
                    except ValueError: # Handle empty int sequence if checks fail
                        if model_instance.verbose: print(f"Warning (fit.py): Could not determine dG for {res_obj.label} due to empty int/offset.")
                
                ctT = first_spec.T
                maxint = 0.0
                if len(first_spec.int) > 0:
                    maxint = np.max(first_spec.int)

                r1 = 1.0
                if maxint > 0 and ctT > 0:
                    r1 = -np.log(maxint) / ctT
                
                r2a = first_spec.initr2a
                p_initial_list.extend([dG, r1, r2a])
    else: # Exchange models (Baldwin, Matrix)
        kex_c = initConf.get('kex', {'min': 10.0, 'max': 400.0, 'nsteps': 6})
        pb_c = initConf.get('pB', {'min': 0.01, 'max': 0.1, 'nsteps': 6})
        
        kex_v_steps = int(kex_c.get('nsteps', 6))
        pB_v_steps = int(pb_c.get('nsteps', 6))

        kex_v = np.linspace(kex_c['min'], kex_c['max'], kex_v_steps)
        pB_v = np.linspace(pb_c['min'], pb_c['max'], pB_v_steps)
        
        minChi2, best_p0_grid = float('inf'), None

        for ikv in kex_v:
            for ipbv in pB_v:
                curr_p_g = []
                kab = ipbv * ikv
                kba = ikv - kab
                if kab < 0 or kba < 0:
                    continue
                curr_p_g.extend([kab, kba])
                
                valid_res_params, tmp_res_p_list = True, []
                for r_o in model_instance.dataset.res:
                    if r_o.active:
                        if not r_o.estSpecs or not r_o.estSpecs[0].int:
                            valid_res_params = False
                            break
                        fs = r_o.estSpecs[0]
                        dG_res = 0.0
                        if len(fs.offset) > 0 and len(fs.int) > 0:
                             try: dG_res = fs.offset[np.argmin(fs.int)]
                             except ValueError: pass # dG_res remains 0.0
                        
                        ctT_res = fs.T
                        mi_res = 0.0
                        if len(fs.int) > 0: mi_res = np.max(fs.int)

                        r1_res = 1.0
                        if mi_res > 0 and ctT_res > 0: r1_res = -np.log(mi_res) / ctT_res
                        
                        tmp_res_p_list.extend([dG_res, fs.initdw, r1_res, fs.initr2a, fs.initr2b])
                
                if not valid_res_params:
                    continue
                
                curr_p_g.extend(tmp_res_p_list)
                
                try:
                    # model_instance.errFunc updates model_instance.chi2
                    model_instance.errFunc(np.array(curr_p_g, dtype=float))
                    current_chi2 = model_instance.chi2
                    if current_chi2 < minChi2:
                        minChi2 = current_chi2
                        best_p0_grid = list(curr_p_g)
                except Exception as e:
                    if model_instance.verbose:
                        print(f"Error in errFunc during grid search for initial params (fit.py): {e}")
                    continue
            
        if best_p0_grid:
            p_initial_list = best_p0_grid
        else:
            if model_instance.verbose: print("Grid search failed for initial params (fit.py), using defaults.")
            default_kex = (kex_c['min'] + kex_c['max']) / 2.0
            default_pB = (pb_c['min'] + pb_c['max']) / 2.0
            p_initial_list.extend([default_pB * default_kex, (1.0 - default_pB) * default_kex])
            for r_o in model_instance.dataset.res:
                if r_o.active:
                    if not r_o.estSpecs or not r_o.estSpecs[0].int:
                        p_initial_list.extend([0.0, 0.1, 1.0, 10.0, 20.0]) # dG, dw, R1, R2a, R2b defaults
                        continue
                    fs = r_o.estSpecs[0]
                    dG_res = 0.0
                    if len(fs.offset) > 0 and len(fs.int) > 0:
                        try: dG_res = fs.offset[np.argmin(fs.int)]
                        except ValueError: pass
                    
                    ctT_res = fs.T
                    mi_res = 0.0
                    if len(fs.int) > 0: mi_res = np.max(fs.int)

                    r1_res = 1.0
                    if mi_res > 0 and ctT_res > 0: r1_res = -np.log(mi_res) / ctT_res
                    p_initial_list.extend([dG_res, fs.initdw, r1_res, fs.initr2a, fs.initr2b])

    if not p_initial_list:
        raise ValueError("Failed to generate any initial parameters (fit.py).")
    
    if model_instance.verbose: print("Initial parameter generation (in fit.py) finished.")
    return np.array(p_initial_list, dtype=float)

def perform_least_squares_fit(model_instance, p0_initial):
    """
    Performs the least squares fitting using scipy.optimize.least_squares.
    model_instance.method must be set before calling this function.
    model_instance.errFunc is used as the objective function.
    """
    if not isinstance(p0_initial, np.ndarray): 
        p0 = np.array(p0_initial, dtype=float)
    else:
        p0 = p0_initial

    bounds_min, bounds_max = [], []
    num_active_res = sum(1 for res_obj in model_instance.dataset.res if res_obj.active)
    expected_len = 0

    if model_instance.method == 'NoEx':
        expected_len = num_active_res * 3
        for _ in range(num_active_res): # dG, R1, R2a
            bounds_min.extend([-np.inf, 0, 0])
            bounds_max.extend([np.inf, np.inf, np.inf])
    else: # Exchange models (Baldwin, Matrix)
        expected_len = 2 + num_active_res * 5
        bounds_min.extend([0, 0]) # kab, kba
        bounds_max.extend([np.inf, np.inf])
        for _ in range(num_active_res): # dG, dw, R1, R2a, R2b
            bounds_min.extend([-np.inf, -np.inf, 0, 0, 0])
            bounds_max.extend([np.inf, np.inf, np.inf, np.inf, np.inf])
    
    if len(p0) != expected_len:
        if model_instance.verbose:
            print(f"Warning (fit.py): p0 length ({len(p0)}) != expected ({expected_len}) for method {model_instance.method}. Adjusting bounds.")
        # Adjust bounds to match p0 length if mismatched
        current_bounds_len = len(bounds_min)
        if len(p0) < current_bounds_len:
            bounds_min = bounds_min[:len(p0)]
            bounds_max = bounds_max[:len(p0)]
        elif len(p0) > current_bounds_len:
            # This case implies p0 is longer than expected based on active residues and method.
            # This could be due to a mismatch in logic or an unexpected p0.
            # For robustness, fill remaining bounds, though this might indicate a problem.
            diff = len(p0) - current_bounds_len
            bounds_min.extend([-np.inf] * diff)
            bounds_max.extend([np.inf] * diff)
            if model_instance.verbose:
                print(f"Warning (fit.py): p0 is longer than expected. Filled {diff} extra bounds with (-inf, inf).")
    
    scipy_bounds = (np.array(bounds_min), np.array(bounds_max))

    if model_instance.verbose:
        print(f"Initiating fit (in fit.py, method: {model_instance.method}): Using scipy.optimize.least_squares with {len(p0)} params.")

    result = None
    p1_optimized = np.copy(p0) 
    covariance_matrix = np.full((len(p0), len(p0)), np.nan)

    try:
        result = least_squares(model_instance.errFunc, x0=p0, bounds=scipy_bounds, method='trf',
                               ftol=1e-9, xtol=1e-9, gtol=1e-9,
                               verbose=2 if model_instance.verbose else 0)
        p1_optimized = result.x
        if result.jac is not None and result.jac.shape[1] == len(p0): # Ensure jacobian dimension matches params
            try:
                jtj = result.jac.T @ result.jac
                if np.linalg.cond(jtj) < 1 / np.finfo(jtj.dtype).eps: # Check condition number before inverting
                    covariance_matrix = np.linalg.inv(jtj)
                else:
                    if model_instance.verbose: print("Jacobian^T * Jacobian is singular or ill-conditioned (fit.py); covariance calculation failed.")
            except np.linalg.LinAlgError:
                if model_instance.verbose: print("Covariance matrix computation failed (singular Jacobian in fit.py).")
            except ValueError as ve: # Catches shape mismatches if any occur before inv
                 if model_instance.verbose: print(f"ValueError during covariance calculation (fit.py): {ve}")
        elif model_instance.verbose:
             details = f"Jacobian shape: {result.jac.shape if result.jac is not None else 'None'}"
             print(f"Jacobian not available or has unexpected dimensions for covariance calculation (fit.py). {details}")
             
    except Exception as e:
        if model_instance.verbose: print(f"Error during least_squares fitting (in fit.py): {e}")

    if model_instance.verbose:
        print("Fit (in fit.py) completed.")
        if result:
            print(f"SciPy Status: {result.status} ({result.message})")
            print(f"Final cost: {result.cost:.4e}, NFEV: {result.nfev}, NJEV: {getattr(result, 'njev', 'N/A')}")
            if not np.allclose(p1_optimized, p0) :
                 print(f"Optimized parameters (first 3): {p1_optimized[:3]}")
            else:
                 print("Parameters did not change from initial guess.")
        else:
            print("Fitting process did not yield a result object.")
            
    return p1_optimized, covariance_matrix
