from matplotlib import pyplot
import numpy as np
import sys
#
#
# def plot_results(filename, results, title, field, ylabel):
#     print(results)
#     x_axis = np.asarray(range(len(results[0][1].all_logs)))
#     plots = {}
#     for r in results:
#         y_values = [l[field] for l in r[1].all_logs]
#         plots[r[0]] = y_values
#     print(plots.items(), x_axis)
#     for k, v in plots.items():
#         pyplot.plot(x_axis, np.asarray(v), 'o-', label=k)
#     pyplot.xlabel('Epoch')
#     pyplot.ylabel(ylabel)
#     pyplot.title(title)
#     pyplot.xticks(np.arange(0, x_axis.shape[0] + 1, 1.0))
#     pyplot.legend(loc='best')
#     pyplot.savefig('reports/' + filename)
#     pyplot.clf()


def plot_results(results_filenames, title, column_names, labels, ylabel, plot_filename=None):
    results_filenames = results_filenames.split(',')
    column_names = [x for x in column_names.split(',')]
    columns = {}
    labels = labels.split(',')
    label_index = 0
    for results_filename in results_filenames:
        name_to_index = {}
        with open('reports/' + results_filename, 'r') as f:
            header = f.readline().split(',')
            for i in range(len(header)):
                name_to_index[header[i].strip()] = i

            rows = filter(None, f.read().split('\n'))
        for i in range(len(rows)):
            rows[i] = rows[i].split(',')
        for cn in column_names:
            ci = name_to_index[cn]
            columns[labels[label_index]] = [float(rows[i][ci]) for i in range(len(rows)) if rows[i]]
            label_index += 1

    plot_filename = plot_filename or (title.lower().replace(' ', '_').replace('.', '') + '.png')
    x_axis = np.asarray(range(max([len(v) for v in columns.values()])))
    for label, values in columns.items():
        pyplot.plot(x_axis, np.asarray(values), 'o-', label=label)

    pyplot.xlabel('Epoch')
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.xticks(np.arange(0, x_axis.shape[0] + 1, 1.0))
    pyplot.legend(loc='best')
    pyplot.savefig('reports/' + plot_filename)
    pyplot.clf()

if __name__ == "__main__":
    print(sys.argv)
    locals()['plot_results'](*sys.argv[1:])