#!/usr/bin/env python

"""
Original by Donghan Lee
Python 3 conversion, initGuessAll removal, scipy.optimize.least_squares integration,
and Parallelization: 2025
"""

import sys
import json
import numpy as np
from estmodel import est_model # Assuming estmodel.py is correctly named and in path
import multiprocessing
import os # For os.cpu_count()
# import time # For potential per-worker seeding if needed

# --- Worker function for parallel MC runs (remains the same as previous version) ---
def run_single_mc_iteration(args_tuple):
    conf, pp_optimized_params_for_mc, worker_id = args_tuple
    mmc = est_model()
    mmc.verbose = False 
    datasets_names = conf['datasets']
    try:
        for dataset_name_mc in datasets_names:
            mmc.dataset.addDataWithError(dataset_name_mc)
        residues_config = conf['residues']
        for r_entry_mc in residues_config:
            resid_label_mc = r_entry_mc['name']
            active_flag_mc = r_entry_mc['flag']
            found_mc = False
            for res_obj_mmc in mmc.dataset.res:
                if res_obj_mmc.label == resid_label_mc:
                    found_mc = True
                    if active_flag_mc == 'on': res_obj_mmc.active = True
                    elif active_flag_mc == 'off': res_obj_mmc.active = False
                    else: return None 
                    break
            if not found_mc: return None
        out_mc_fit = mmc.fit(p0=pp_optimized_params_for_mc, fitting_config=conf['init'])
        return out_mc_fit[0]
    except Exception as e:
        # print(f"[Worker {worker_id}] Error: {e}", file=sys.stderr) # Optional: log worker errors
        return None

# --- Main script execution ---
def main():
    if len(sys.argv) < 3:
        sys.stderr.write(f"Usage: {sys.argv[0]} config_file.json number_of_mc_runs [num_processes]\n")
        sys.stderr.write("  [num_processes] is optional, defaults to number of CPU cores.\n")
        sys.exit(1)

    config_file_path = sys.argv[1]
    try:
        nrun = int(sys.argv[2])
        if nrun <= 0: raise ValueError("Number of MC runs must be positive.")
    except ValueError as e:
        sys.stderr.write(f"Error: Invalid number of MC runs '{sys.argv[2]}'. {e}\n"); sys.exit(1)

    num_processes = None 
    if len(sys.argv) > 3:
        try:
            num_processes = int(sys.argv[3])
            if num_processes <= 0: raise ValueError("Number of processes must be positive.")
        except ValueError as e:
            sys.stderr.write(f"Error: Invalid num_processes '{sys.argv[3]}'. {e}\nUsing default.\n"); num_processes = None

    try:
        with open(config_file_path, 'r') as cf: conf = json.load(cf)
    except FileNotFoundError:
        sys.stderr.write(f"Error: Config file not found: {config_file_path}\n"); sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON in config file: {config_file_path}\n"); sys.exit(1)

    m2 = est_model()
    m2.verbose = True 
    print('**************\n' + m2.programName + ' - MC Run Setup (Parallelized)\n**************')
    project_name = conf['Project Name']; datasets_names = conf['datasets']
    for dataset_name in datasets_names: m2.dataset.addData(dataset_name)
    residues_config = conf['residues']
    for r_entry in residues_config:
        resid_label, active_flag = r_entry['name'], r_entry['flag']
        if m2.verbose: print(f'Configuring Residue for initial fit: {resid_label} {active_flag}')
        found = False
        for res_obj_m2 in m2.dataset.res:
            if res_obj_m2.label == resid_label:
                found = True
                if active_flag == 'on': res_obj_m2.active = True
                elif active_flag == 'off': res_obj_m2.active = False
                else: sys.stderr.write(f"Error: Wrong flag for {resid_label}\n"); sys.exit(1)
                break
        if not found: sys.stderr.write(f"Error: Residue {resid_label} not found.\n"); sys.exit(1)

    if m2.verbose: print(f"Generating data PDF: {project_name}_data.pdf")
    m2.datapdf(f"{project_name}_data.pdf")
    if m2.verbose: print("Performing initial fit...")
    out_initial_fit = m2.fit(fitting_config=conf['init'])
    pp_optimized_params = out_initial_fit[0]
    if m2.verbose: print(f"Generating PDF for initial fit: {project_name}.pdf")
    m2.pdf(pp_optimized_params, f"{project_name}.pdf")

    tasks_args = [(conf, pp_optimized_params, i) for i in range(nrun)]
    if num_processes is None: num_processes = os.cpu_count()
    print(f"\nStarting {nrun} Monte Carlo runs in parallel using {num_processes} CPU cores...")

    with multiprocessing.Pool(processes=num_processes) as pool:
        results_from_pool = pool.map(run_single_mc_iteration, tasks_args)

    all_mc_fitted_params_list = [res for res in results_from_pool if res is not None]
    successful_runs = len(all_mc_fitted_params_list)
    failed_runs = nrun - successful_runs
    print(f"Completed {successful_runs}/{nrun} MC runs successfully.")
    if failed_runs > 0: print(f"Warning: {failed_runs} MC runs failed and were excluded.")
    if not all_mc_fitted_params_list: print("No MC runs successful. Exiting."); sys.exit(1)

    if m2.verbose: print("\nGenerating MC log buffer...")
    
    # --- MODIFICATION HERE ---
    # Adjust to expect 2 return values from getLogBufferMC
    # This assumes getLogBufferMC internally calculates and uses std_dev for its string output,
    # but only returns the log string and mean parameters.
    log_buffer_content_mc, mean_mc_params = m2.getLogBufferMC(
        pp_optimized_params, all_mc_fitted_params_list
    )

    # If you need std_dev_mc_params explicitly in mcrun.py, calculate it here:
    std_dev_mc_params_explicit = np.std(np.array(all_mc_fitted_params_list), axis=0)
    # --- END MODIFICATION ---

    mc_result_filename = f"{project_name}_mc.txt"
    if m2.verbose: print(f"Saving MC results to: {mc_result_filename}")
    with open(mc_result_filename, 'w') as mc_file: mc_file.write(log_buffer_content_mc)

    if m2.verbose: print(f"Generating PDF for mean MC parameters: {project_name}_mcmean.pdf")
    m2.pdf(mean_mc_params, f"{project_name}_mcmean.pdf")

    if m2.verbose:
        print("\n--- Monte Carlo Parameter Statistics (from mcrun.py) ---")
        print(f"Number of successful MC runs: {len(all_mc_fitted_params_list)}")
        num_params_to_show = min(len(mean_mc_params), 5)
        print("Mean Parameters (from MC) +/- StdDev (from MC, calculated in mcrun.py):")
        for i in range(num_params_to_show):
            # Ensure std_dev_mc_params_explicit has content for this index
            std_dev_val = std_dev_mc_params_explicit[i] if i < len(std_dev_mc_params_explicit) else np.nan
            print(f"  Param {i}: {mean_mc_params[i]:.4f} +/- {std_dev_val:.4f}")
        
    print('########\nMC Run script finished (Parallelized).\n')

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main()
