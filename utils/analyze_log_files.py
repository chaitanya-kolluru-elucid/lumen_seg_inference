import matplotlib.pyplot as plt
import os
import distinctipy

def plot_scatter(dictionary):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Get the keys and number of keys
    keys = list(dictionary.keys())
    num_keys = len(keys)

    # Set up colors
    colors = distinctipy.get_colors(len(dictionary[keys[0]]), rng=10)
                                 
    # Plot each key's values in a single column
    for i, key in enumerate(keys):
        values = dictionary[key]
        x = [i] * len(values)
        plt.scatter(x, values, color=colors, label = keys[i])

    plt.xticks(range(len(keys)), labels=keys, fontsize=9, rotation=90)
    plt.ylabel('Time [seconds]')
    plt.xlabel('Steps in inferencing')

    # Show the plot
    plt.show()

if __name__ == '__main__':

    inference_log_filepath = os.path.join('./logs', '16-Apr-24-20:16:52-run_inference_pipeline.log')
    memory_usage_log_filepath = os.path.join('./logs', '16-Apr-24-20:16:54-gpu_mem_track.log')

    case_names = []
    inference_times = []
    predictions_argmax_times = []
    case_processing_times = []

    steps = ['Inferencing', 'Predictions arg max', 'Overall processing time']
    timing_dict = {}

    for step in steps:
        timing_dict[step] = []

    with open(inference_log_filepath, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):

        if 'Processing case:' in line:

            step_index = 0

            timing_dict[steps[step_index]].append(int(lines[i+1].split('Inference took ')[-1].split(' seconds')[0]))
            timing_dict[steps[step_index + 1]].append(int(lines[i+3].split('Predictions argmax took ')[-1].split(' seconds')[0]))
            timing_dict[steps[step_index + 2]].append(float(lines[i+5].split('Processing this case took ')[-1].split(' seconds')[0]))
            
            case_names.append(line.split('Processing case: ')[-1])

    plot_scatter(timing_dict)
    plt.savefig('Timing info - inference.png', bbox_inches="tight")

    