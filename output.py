#!/usr/bin/env python
# output.py
# Contains output generation logic (PDFs, text logs) extracted from estmodel.py
# Original estmodel.py by Donghan Lee 2015
# Refactoring: 2025

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import plot, errorbar, figure, title, legend, xlabel, ylabel, grid, xlim, ylim, close
#from os import getlogin, name   #for linux
from os import getlogin
from platform import uname      #for window
from time import ctime

def generate_results_pdf(model_instance, p1_flat, pdf_file_name):
    """
    Generates a PDF report of the fitting results.
    Corresponds to the original est_model.pdf() method.
    """
    if not isinstance(p1_flat, np.ndarray):
        p1_flat = np.array(p1_flat, dtype=float)

    pdf = PdfPages(pdf_file_name)
    try:
        # seParam is a method of model_instance
        par_struc = model_instance.seParam(p1_flat)
    except Exception as e:
        print(f"Error in seParam during PDF generation for {pdf_file_name}: {e}. Skipping PDF generation.")
        if model_instance.verbose:
            print(f"Parameters causing seParam error in PDF: {p1_flat}")
        pdf.close()
        return

    kab, kba, dGs, dws, r1s, r2as, r2bs = (None,) * 7

    if model_instance.method == 'NoEx':
        if len(par_struc) == 3:
            dGs, r1s, r2as = par_struc[0], par_struc[1], par_struc[2]
        else:
            print(f"Warning: Parameter structure mismatch for NoEx in PDF {pdf_file_name}. Got {len(par_struc)} elements. Expected 3.")
            pdf.close()
            return
    else:
        if len(par_struc) == 7:
            kab, kba = par_struc[0], par_struc[1]
            dGs, dws, r1s, r2as, r2bs = par_struc[2], par_struc[3], par_struc[4], par_struc[5], par_struc[6]
        else:
            print(f"Warning: Parameter structure mismatch for {model_instance.method} in PDF {pdf_file_name}. Got {len(par_struc)} elements. Expected 7.")
            pdf.close()
            return

    for i, res_obj in enumerate(model_instance.dataset.res):
        if res_obj.active:
            if not res_obj.estSpecs:
                if model_instance.verbose:
                    print(f"No spectra for active residue {res_obj.label}, skipping PDF page in {pdf_file_name}.")
                continue

            fig = figure(figsize=(8.5, 6))
            ax = fig.add_subplot(1, 1, 1)

            colorsSet = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            all_offsets_for_res, all_intensities_for_res, all_stddevs_for_res = [], [], []

            for j, ep in enumerate(res_obj.estSpecs):
                if not ep.offset:
                    if model_instance.verbose:
                        print(f"Empty EstSpec {j} for residue {res_obj.label}, skipping in {pdf_file_name}.")
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
                    # Access calculation methods from model_instance
                    if model_instance.method == 'Matrix':
                        calc_val = model_instance.matrix_calc(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                    elif model_instance.method == 'Baldwin':
                        calc_val = model_instance.Baldwin(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                    elif model_instance.method == 'NoEx':
                        calc_val = model_instance.matrix_calc_noex(dGs[i],r1s[i],r2as[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
                    else: # Fallback or should raise error if method is unknown
                        calc_val = model_instance.Baldwin(kab,kba,dGs[i],dws[i],r1s[i],r2as[i],r2bs[i],offset_val_plot,ep.T,ep.v1,ep.v1err,ep.field)
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
            if all_intensities_for_res and all_stddevs_for_res:
                min_y_val = np.min(np.array(all_intensities_for_res) - np.array(all_stddevs_for_res))
                max_y_val = np.max(np.array(all_intensities_for_res) + np.array(all_stddevs_for_res))
                plot_min_y = min(0.0, min_y_val * 0.95 if min_y_val >= 0 else min_y_val * 1.05)
                plot_max_y = max_y_val * 1.05 if max_y_val >= 0 else max_y_val * 0.95
                if plot_max_y <= plot_min_y: plot_max_y = plot_min_y + 0.1
                ax.set_ylim(plot_min_y, plot_max_y)

            fig.tight_layout(rect=[0, 0, 0.85, 1])
            pdf.savefig(fig)
            close(fig)
    pdf.close()
    if model_instance.verbose:
        print(f"Generated results PDF: {pdf_file_name}")

def generate_data_pdf(model_instance, pdf_file_name):
    """
    Generates a PDF showing the input experimental data.
    Corresponds to the original est_model.datapdf() method.
    """
    pdf = PdfPages(pdf_file_name)
    for r_o in model_instance.dataset.res:
        if r_o.active:
            fig=figure(figsize=(8.5,6));ax=fig.add_subplot(1,1,1)
            cs=['b','g','r','c','m','y','k'];mxv,miv,mayv,miyv=[],[],[],[]
            for ispec,ep in enumerate(r_o.estSpecs):
                if not ep.offset: continue
                mxv.append(np.min(ep.offset));miv.append(np.max(ep.offset))
                c=cs[ispec%len(cs)]
                ax.errorbar(ep.offset,ep.int,yerr=ep.intstd,fmt=f'{c}o',ms=3,label=f'exp {ep.v1:.1f}Hz {ep.T*1e3:.1f}ms')
                if ep.int and ep.intstd:
                    miyv.append(np.min(np.array(ep.int)-np.array(ep.intstd)));mayv.append(np.max(np.array(ep.int)+np.array(ep.intstd)))
            
            ax.set_title(r_o.label);ax.set_xlabel('Chemical Shift Offset (ppm)');ax.set_ylabel('Intensity')
            box=ax.get_position();ax.set_position([box.x0,box.y0,box.width*0.75,box.height])
            ax.legend(loc='center left',bbox_to_anchor=(1,0.5), fontsize='small');ax.grid(True)
            
            plot_min_y_data, plot_max_y_data = 0, 0.1
            if miyv and mayv:
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
    if model_instance.verbose:
        print(f"Generated data PDF: {pdf_file_name}")

def generate_log_buffer(model_instance, fit_output_tuple):
    """
    Generates a string buffer with fitting log information.
    Corresponds to the original est_model.getLogBuffer() method.
    """
    p_optimized, covar = fit_output_tuple[0], fit_output_tuple[1]
    parstd_diag = np.full_like(p_optimized, np.nan)
    if covar is not None and covar.shape == (len(p_optimized), len(p_optimized)) and not np.all(np.isnan(covar)):
        diag_covar = np.diag(covar)
        valid_indices = ~np.isnan(diag_covar) & ~np.isinf(diag_covar) & (diag_covar >= 0)
        parstd_diag[valid_indices] = np.sqrt(diag_covar[valid_indices])
        if np.any(~valid_indices & (~np.isnan(diag_covar))):
             if model_instance.verbose: print("Warning (output.py/log): Some diagonal covar elements invalid; std dev set to NaN.")
    elif model_instance.verbose: print("Warning (output.py/log): Covar matrix issue; std devs are NaN.")

    log_list=['*'*51, model_instance.programName, '*'*51+'\n']
    try:
        user, hostname = getlogin(), uname()[1]
        log_list.append(f'User: {user}@{hostname}')
    except Exception:
        log_list.append('User: Unknown')
    log_list.extend([f'{ctime()}', '*'*51])
    
    # Call seParam on the model_instance
    par_struc = model_instance.seParam(p_optimized)
    kab_v, kba_v, dGs, dws, r1s, r2as, r2bs = (None,)*7
    if model_instance.method=='NoEx':
        dGs,r1s,r2as=par_struc[0],par_struc[1],par_struc[2]
    else:
        kab_v,kba_v=par_struc[0],par_struc[1]
        dGs,dws,r1s,r2as,r2bs=par_struc[2],par_struc[3],par_struc[4],par_struc[5],par_struc[6]

    for ir,ro in enumerate(model_instance.dataset.res):
        if ro.active:
            log_list.append(f'\n# {ro.label}')
            for es in ro.estSpecs:
                log_list.extend([f'B0 [MHz]: {es.field:8.3f}',f'T [ms]: {es.T*1e3:8.3f}',f'v1 [Hz]: {es.v1:8.3f}'])
                log_list.append(f"{'Offset':>8} {'ExpI':>12} {'ExpStd':>12} {'CalcI':>12}")
                for ko,ovl in enumerate(es.offset):
                    ec=0.0
                    if model_instance.method=='Matrix':ec=model_instance.matrix_calc(kab_v,kba_v,dGs[ir],dws[ir],r1s[ir],r2as[ir],r2bs[ir],ovl,es.T,es.v1,es.v1err,es.field)
                    elif model_instance.method=='Baldwin':ec=model_instance.Baldwin(kab_v,kba_v,dGs[ir],dws[ir],r1s[ir],r2as[ir],r2bs[ir],ovl,es.T,es.v1,es.v1err,es.field)
                    elif model_instance.method=='NoEx':ec=model_instance.matrix_calc_noex(dGs[ir],r1s[ir],r2as[ir],ovl,es.T,es.v1,es.v1err,es.field)
                    ec = np.nan_to_num(ec)
                    log_list.append(f'{ovl:8.3f} {es.int[ko]:12.3f} {es.intstd[ko]:12.3f} {ec:12.3f}')
            log_list.append('*'*51)
    
    # Recalculate chi2, dof etc with optimized params using model_instance.errFunc
    # This ensures model_instance's internal chi2, dof, nvar, npar are updated for the log.
    _ = model_instance.errFunc(p_optimized)
    log_list.extend(['\n\n\n'+'*'*41,f'Results using {model_instance.method} method\n',
                     f'Chi2:     {model_instance.chi2:8.6f}',
                     f'red_chi2: {model_instance.chi2/model_instance.dof if model_instance.dof>0 else float("inf"):8.6f}',
                     f'dof:      {model_instance.dof:d}',f'nv:       {model_instance.nvar:d}',f'np:       {model_instance.npar:d}','*'*43,
                     'Fitted Parameters (+/- StdDev if available):'])
    pi_log=0
    if model_instance.method!='NoEx':
        log_list.append(f'kab [s-1]: {p_optimized[pi_log]:8.3f} +/- {parstd_diag[pi_log]:8.3f}');pi_log+=1
        log_list.append(f'kba [s-1]: {p_optimized[pi_log]:8.3f} +/- {parstd_diag[pi_log]:8.3f}');pi_log+=1
    
    for idx_res_log, ro_log in enumerate(model_instance.dataset.res):
        if ro_log.active:
            log_list.extend(['*'*43,f'residue: {ro_log.label}'])
            if model_instance.method=='NoEx':
                if pi_log + 2 < len(p_optimized):
                    log_list.append(f'peak pos [ppm]: {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'R1 [s-1]:       {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'R2a [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                else: log_list.append("Error (output.py/log): Parameter index out of bounds for NoEx.")
            else: # Exchange models
                if pi_log + 4 < len(p_optimized):
                    log_list.append(f'peak pos [ppm]: {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'cs diff [ppm]:  {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'R1 [s-1]:       {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'R2a [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                    log_list.append(f'R2b [s-1]:      {p_optimized[pi_log]:8.4f} +/- {parstd_diag[pi_log]:8.4f}');pi_log+=1
                else: log_list.append("Error (output.py/log): Parameter index out of bounds for Exchange model.")
    log_list.append('*'*43)
    return "\n".join(log_list)

def generate_mc_log_buffer(model_instance, p_center_initial_fit, all_mc_ps_list):
    """
    Generates a string buffer with Monte Carlo analysis log information.
    Corresponds to the original est_model.getLogBufferMC() method.
    Returns the log string and the mean MC parameters.
    """
    if not all_mc_ps_list:
        log_list=['*'*51,f'{model_instance.programName} - Monte Carlo Analysis - NO SUCCESSFUL RUNS','*'*51+'\n']
        return "\n".join(log_list), np.array([])

    all_mc_ps_array = np.array(all_mc_ps_list)
    mean_mc_params = np.mean(all_mc_ps_array, axis=0)
    std_dev_mc_params = np.std(all_mc_ps_array, axis=0)
    
    log_list=['*'*51,f'{model_instance.programName} - Monte Carlo Analysis','*'*51+'\n']
    try:
        user, hostname = getlogin(), uname()[1]
        log_list.append(f'User: {user}@{hostname}')
    except Exception:
        log_list.append('User: Unknown')
    log_list.extend([f'{ctime()}',f'Num MC runs: {len(all_mc_ps_list)}','*'*51])
    
    # model_instance.method should be set from the initial fit context
    if not model_instance.method:
        print("Warning (output.py/mc_log): model_instance.method not set. Defaulting to Baldwin for seParam.")
        # This is a fallback; ideally, the method is already correctly set on model_instance
        model_instance.method = 'Baldwin' 

    par_s_mc = model_instance.seParam(mean_mc_params) # Uses model_instance.method
    kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m = (None,)*7

    if model_instance.method=='NoEx':
        if len(par_s_mc) == 3: dGs_m,r1s_m,r2as_m=par_s_mc[0],par_s_mc[1],par_s_mc[2]
        else: log_list.append("Error (output.py/mc_log): Param structure mismatch for NoEx.");
    else: # Exchange models
        if len(par_s_mc) == 7: 
            kab_m,kba_m=par_s_mc[0],par_s_mc[1]
            dGs_m,dws_m,r1s_m,r2as_m,r2bs_m=par_s_mc[2],par_s_mc[3],par_s_mc[4],par_s_mc[5],par_s_mc[6]
        else: log_list.append(f"Error (output.py/mc_log): Param structure mismatch for {model_instance.method}.");

    for ir,ro in enumerate(model_instance.dataset.res):
        if ro.active:
            log_list.append(f'\n# {ro.label}')
            for es in ro.estSpecs:
                log_list.extend([f'B0 [MHz]: {es.field:8.3f}',f'T [ms]: {es.T*1e3:8.3f}',f'v1 [Hz]: {es.v1:8.3f}'])
                log_list.append(f"{'Offset':>8} {'ExpI':>12} {'ExpStd':>12} {'CalcI_MC_Mean':>18}")
                for ko,ovl in enumerate(es.offset):
                    ec_m=0.0
                    # Check if params were successfully unpacked for the current method
                    calc_successful = False
                    if model_instance.method=='Matrix' and all(p is not None for p in [kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m]):
                        if ir < len(dGs_m): # Ensure index is valid
                            ec_m=model_instance.matrix_calc(kab_m,kba_m,dGs_m[ir],dws_m[ir],r1s_m[ir],r2as_m[ir],r2bs_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                            calc_successful = True
                    elif model_instance.method=='Baldwin' and all(p is not None for p in [kab_m,kba_m,dGs_m,dws_m,r1s_m,r2as_m,r2bs_m]):
                        if ir < len(dGs_m):
                            ec_m=model_instance.Baldwin(kab_m,kba_m,dGs_m[ir],dws_m[ir],r1s_m[ir],r2as_m[ir],r2bs_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                            calc_successful = True
                    elif model_instance.method=='NoEx' and all(p is not None for p in [dGs_m,r1s_m,r2as_m]):
                        if ir < len(dGs_m):
                            ec_m=model_instance.matrix_calc_noex(dGs_m[ir],r1s_m[ir],r2as_m[ir],ovl,es.T,es.v1,es.v1err,es.field)
                            calc_successful = True
                    
                    if not calc_successful and model_instance.verbose:
                        log_list.append(f"Note: Calc skipped for {ro.label} offset {ovl} due to param unpack issue.")

                    ec_m = np.nan_to_num(ec_m)
                    log_list.append(f'{ovl:8.3f} {es.int[ko]:12.3f} {es.intstd[ko]:12.3f} {ec_m:18.3f}')
            log_list.append('*'*51)
    
    # Use p_center_initial_fit to update model_instance's chi2, dof for the initial fit stats
    _ = model_instance.errFunc(p_center_initial_fit)
    log_list.extend(['\n\n\n'+'*'*41,f'Results using {model_instance.method} (MC Stats)\n',
                     f'Chi2 (init fit): {model_instance.chi2:8.6f}',
                     f'red_chi2 (init fit): {model_instance.chi2/model_instance.dof if model_instance.dof>0 else float("inf"):8.6f}',
                     f'dof (init fit):      {model_instance.dof:d}',f'nv (init fit):       {model_instance.nvar:d}',
                     f'np (init fit):       {model_instance.npar:d}','*'*43,
                     'Fitted Params (Mean from MC +/- StdDev from MC):'])
    pi_log=0
    
    # Check array lengths before indexing
    expected_params_len = 0
    active_res_count = sum(1 for r_obj in model_instance.dataset.res if r_obj.active)
    if model_instance.method != 'NoEx': expected_params_len = 2 + active_res_count * 5
    else: expected_params_len = active_res_count * 3

    if len(mean_mc_params) != expected_params_len or len(std_dev_mc_params) != expected_params_len:
        log_list.append(f"Error (output.py/mc_log): MC parameter array length mismatch. Mean: {len(mean_mc_params)}, StdDev: {len(std_dev_mc_params)}, Expected: {expected_params_len}")
    else:
        if model_instance.method!='NoEx':
            log_list.append(f'kab [s-1]: {mean_mc_params[pi_log]:8.3f} +/- {std_dev_mc_params[pi_log]:8.3f}');pi_log+=1
            log_list.append(f'kba [s-1]: {mean_mc_params[pi_log]:8.3f} +/- {std_dev_mc_params[pi_log]:8.3f}');pi_log+=1
        
        for ro_log in model_instance.dataset.res:
            if ro_log.active:
                log_list.extend(['*'*43,f'residue: {ro_log.label}'])
                if model_instance.method=='NoEx':
                    if pi_log + 2 < len(mean_mc_params): # Check bounds
                        log_list.append(f'peak pos [ppm]: {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R1 [s-1]:       {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2a [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                    else: log_list.append("Error (output.py/mc_log): MC Param index out of bounds for NoEx.")
                else: # Exchange models
                    if pi_log + 4 < len(mean_mc_params): # Check bounds
                        log_list.append(f'peak pos [ppm]: {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'cs diff [ppm]:  {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R1 [s-1]:       {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2a [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                        log_list.append(f'R2b [s-1]:      {mean_mc_params[pi_log]:8.4f} +/- {std_dev_mc_params[pi_log]:8.4f}');pi_log+=1
                    else: log_list.append("Error (output.py/mc_log): MC Param index out of bounds for Exchange model.")
    log_list.append('*'*43)
    return "\n".join(log_list), mean_mc_params

