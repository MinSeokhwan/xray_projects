import os, sys
import numpy as np
sys.path.insert(0, '../')
import analysis.read_simulation as rs
os.environ['DISPLAY'] = 'localhost:10.0'
# Run the script from the build directory
import concurrent.futures
import multiprocessing
import uuid

def generate_unique_string():
    # Generate a UUID4, which is a randomly generated unique identifier
    unique_id = uuid.uuid4()
    # Convert the UUID to a string
    unique_string = str(unique_id)
    return unique_string

def parallel_for_loop(func, iterable):
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=(multiprocessing.cpu_count() - 1)) as executor:
        # Submit each item from the iterable to the executor
        future_to_item = {executor.submit(func, item): item for item in iterable}

        # Retrieve results as they complete
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                result = future.result()
                results[item] = result
            except Exception as e:
                print(f"Error processing item {item}: {e}")
    results_list = []
    for item in iterable:
        results_list.append(results[item])
    return results_list

def simulate_scenario(scenario):
    unique_name = generate_unique_string()
    run_mac_name = 'run_' + unique_name + '.mac'
    root_file_name = 'output_' + unique_name + '.root'
    scenario = '/system/root_file_name ' + root_file_name + '\n' + scenario
    with open(run_mac_name, 'w') as f:
        f.writelines(scenario)
    os.system('./NS ' + run_mac_name)
    _, _, photonsZ = rs.read_simulation(root_file_name, property='fType', key='lepton')
    os.remove(run_mac_name)
    os.remove(root_file_name)
    return len(photonsZ)

def simulate_scenario_field(scenario, property, key, field):
    unique_name = generate_unique_string()
    run_mac_name = 'run_' + unique_name + '.mac'
    root_file_name = 'output_' + unique_name + '.root'
    scenario = '/system/root_file_name ' + root_file_name + '\n' + scenario
    with open(run_mac_name, 'w') as f:
        f.writelines(scenario)
    os.system('./NS ' + run_mac_name)
    photonsField = rs.read_simulation_field(root_file_name, property=property, key=key, field=field)
    os.remove(run_mac_name)
    os.remove(root_file_name)
    return photonsField

def simulate_scenario_field_multiple(scenario, nEvents, i):
    import shutil
    if os.path.isdir(str(i)):
        shutil.rmtree(str(i))
    unique_name = generate_unique_string()
    run_mac_name = 'run_' + unique_name + '.mac'
    with open(run_mac_name, 'w') as f:
        f.writelines(scenario)
    os.system('./NS ' + run_mac_name + ' ' + str(int(nEvents)) + ' ' + str(i))
    os.remove(run_mac_name)

def fit_exp_param(t, x, verbose=False):
    import scipy.optimize as so
    fit = so.curve_fit(lambda t_tag,a,b,c: a*np.exp(-b*t_tag)+c,  t,  x)
    a_fit, b_fit, c_fit = fit[0][0], fit[0][1], fit[0][2]
    fitted_decay_param = 1 / b_fit

    if verbose:
        import matplotlib.pyplot as plt
        print('Fitting formula: a * exp(-b*t) + c')
        print('fit result: a = {}, b = {}, c = {}'.format(a_fit, b_fit, c_fit))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, x)
        t_tag = np.linspace(0., t.max(), 101)
        t_exp = a_fit*np.exp(-t_tag/fitted_decay_param) + c_fit
        ax.plot(t_tag, t_exp)
        plt.show()

    return fitted_decay_param