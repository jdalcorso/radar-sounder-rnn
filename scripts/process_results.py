import numpy as np
import argparse
'''
This script processes the results of a an experiment launched via
experiment.sh by computing mean and standard deviation across
different seeds.
The file is expected to be called latest_experiment.sh
'''

def create_parser():
    parser = argparse.ArgumentParser(description="Process an experiment log.")
    parser.add_argument('-f', '--filepath', type=str, default = '/latest_experiment.txt')
    return parser

def main(args):
    N = 0 # Number of seeds
    list_of_reports = []
    list_of_oas = []
    with open(args.filepath,'r') as file:
        for line in file:
            if line.startswith('{'):
                line_dict = eval(line)
                report = np.zeros((4,3)) # Class, Metric
                for class_idx, item in enumerate([str(i) for i in range(4)]):
                    for metric_idx, subitem in enumerate(list(line_dict[item].keys())[:-1]):
                        report[class_idx, metric_idx] = line_dict[item][subitem]
                list_of_oas.append(line_dict['accuracy'])
                list_of_reports.append(report)
                N += 1
    array_of_reports = np.stack(list_of_reports) # NCM
    array_of_oas = np.array(list_of_oas) # N

    print('Mean and Std of Metrics over {} seeds'.format(str(N)))
    print(array_of_reports.mean(0))
    print(array_of_reports.std(0))
    print('Mean and Std of Overall Accuracy over {} seeds'.format(str(N)))
    print(array_of_oas.mean())
    print(array_of_oas.std())

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    main(args)

