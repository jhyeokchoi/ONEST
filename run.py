#!/usr/bin/env python

"""
Original written by Donghan Lee 2025
Python 3 conversion, initGuessAll removal, and scipy.optimize.least_squares integration: 2025
"""

import sys
import json
from estmodel import est_model # Ensure estmodel.py (modified est_model_noex.py) is in path

def load_config(config_file_path):
    """설정 파일을 로드하고 검증합니다."""
    try:
        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)
            if not all(key in config for key in ['Project Name', 'datasets', 'residues', 'init']):
                raise ValueError("Invalid config file structure.")
            return config
    except FileNotFoundError:
        sys.stderr.write(f"Error: Config file not found: {config_file_path}\n")
        sys.exit(1)
    except json.JSONDecodeError:
        sys.stderr.write(f"Error: Invalid JSON format in config file: {config_file_path}\n")
        sys.exit(1)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

def process_residues(model_instance, residues_config):
    """residue 데이터를 처리하여 모델 내 데이터셋의 잔기 활성 상태를 설정합니다."""
    for res_entry in residues_config:
        residue_name = res_entry['name']
        active_flag = res_entry['flag']
        if model_instance.verbose:
            print(f"Configuring Residue: {residue_name}, Active: {active_flag}")
        
        found_res = False
        for res_obj in model_instance.dataset.res:
            if res_obj.label == residue_name:
                found_res = True
                if active_flag == 'on':
                    res_obj.active = True
                elif active_flag == 'off':
                    res_obj.active = False
                else:
                    sys.stderr.write(f"Error: Wrong flag '{active_flag}' for residue {residue_name}\n")
                    sys.exit(1)
                break
        if not found_res:
            sys.stderr.write(f"Error: Residue {residue_name} not found in dataset.\n")
            sys.exit(1)

def main():
    """메인 함수입니다."""
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} config_file.json\n")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config = load_config(config_file_path)

    model = est_model()
    model.verbose = True # Set verbosity as needed

    print("**************")
    print(model.programName)
    print("**************")

    for dataset_name in config['datasets']:
        if model.verbose: print(f"Loading dataset: {dataset_name}")
        model.dataset.addData(dataset_name)

    process_residues(model, config['residues'])

    project_name = config['Project Name']
    if model.verbose: print(f"Generating data PDF: {project_name}_data.pdf")
    model.datapdf(f"{project_name}_data.pdf")

    # initGuessAll call is removed.
    # model.fit now takes fitting_config (which is config['init']) to generate initial parameters.
    if model.verbose: print("Starting model fitting...")
    fit_result_tuple = model.fit(fitting_config=config['init']) # Pass config['init']
    
    optimized_params = fit_result_tuple[0] 

    if model.verbose: print("Generating log buffer...")
    log_buffer_content = model.getLogBuffer(fit_result_tuple)

    result_file_name = f"{project_name}_result.txt"
    if model.verbose: print(f"Saving results to: {result_file_name}")
    with open(result_file_name, 'w') as result_file:
        result_file.write(log_buffer_content)
    # Print log to console as well, if desired (original did this for the older run.py)
    # print(f"\nFull log output:\n{log_buffer_content}") 

    if model.verbose: print(f"Generating results PDF: {project_name}.pdf")
    model.pdf(optimized_params, f"{project_name}.pdf")

    print("########\nRun script finished.\n")

if __name__ == "__main__":
    main()
