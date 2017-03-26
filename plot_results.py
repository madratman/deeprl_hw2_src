import os
import matplotlib.pyplot as plt

results_dir = '/home/ratneshmadaan/deeprl_results'
type_of_Q_learning = os.listdir(results_dir)

for each_type in type_of_Q_learning:
	if 'py' in each_type:
		continue
	each_experiment = os.listdir(os.path.join(results_dir, each_type))
	for trial in each_experiment:
		metrics = os.listdir(os.path.join(results_dir, each_type, trial))
		for each_metric in metrics:
			if 'png' in each_metric:
				continue
			print os.path.join(results_dir, each_type, trial, each_metric)
			file = open(os.path.join(results_dir, each_type, trial, each_metric), 'r')
			stuff = file.read().split('\n')
			if stuff[-1] == '':
				del stuff[-1]
			plt.figure()
			stuff_float = [float(thing) for thing in stuff]
			plt.plot(range(len(stuff_float)), stuff_float)
			filename = each_metric.split('.')[0] + '.png'
			plt.savefig(os.path.join(results_dir, each_type, trial, filename))


