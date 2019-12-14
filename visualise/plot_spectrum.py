import numpy as np
import matplotlib.pyplot as plt


def plot_spectrum(mode, *args, **kwargs):
    if mode == 'lanczos':
        plot_spectrum_lanczos(*args, **kwargs)
    else:
        raise ValueError('Mode' + str(mode) + " is not understood.")


def plot_spectrum_lanczos(
        result: dict = None,
        path: str = None,
        display_spectrum_stats: bool = True,
):
    """
    Generate a stem plot of the eigenspectrum, if we are using Lanczos
    Parameters
    ----------
    result: dict: the return values from core/spectrum
    path: str: the path string to the saved spectrum result
    display_spectrum_stats: if True, a set of on-screen statistics of the eigenspectrum will be displayed

    Returns
    -------

    """
    if result is not None:
        a = result
    elif path is not None:
        a = np.load(path)
    else: raise ValueError('Either result or path needs to be non-empty.')
    eig = []
    weight = []
    for i in range(0, len(a['eigvals'])):
        eig.append(a['eigvals'][i, 0])
        weight.append(a['gammas'][i])
    markerline, stemlines, baseline = plt.stem(eig, weight, '-', linefmt='black')
    plt.xlabel('Eigenvalue Size')
    plt.ylabel('Spectral Density')

    # setting property of baseline with color red and linewidth 2
    plt.yscale('log')
    # plt.xscale('symlog')
    plt.xscale('linear')
    plt.setp(baseline, color='r', linewidth=2)
    plt.rcParams["figure.figsize"] = (10, 3)
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', titlesize=16)
    plt.rc('xtick', labelsize=16)

    plt.xticks(np.arange(min(eig), max(eig), (max(eig) - min(eig)) / 3))
    plt.xticks(list(plt.xticks()[0]) + [max(eig)])
    plt.tick_params(labelbottom='on', labeltop='off')

    if display_spectrum_stats:
        print('\n Spectral Statistics')
        print('Maximum Value is ' + str(max(eig)))
        print('Minimum Value is ' + str(min(eig)))
        print('Mean of Bulk is ' + str(np.median(eig)))
        print('number of negative eigenvalues')

        negeigs = 0
        negweight = 0
        for i in range(0, len(eig)):
            if eig[i] < 0:
                negeigs = negeigs + 1
                #             print('eig val')
                #             print(eig[i])
                #             print('weight val')
                #             print(weight[i])
                if weight[i] < 0.6:
                    negweight = negweight + weight[i]
        print('number of negative Ritz values')
        print(negeigs)
        print('weight of negative Ritz values')
        print(negweight)
        print('pseudo log determinant')
        print(np.sum(np.log(np.abs(eig)) * weight))
        print('trace')
        print(np.dot(eig, weight))
        print('\n')
        idx = np.argmax(weight)
        weightvalue = weight[np.argmax(weight)]
        print('degeneracy value = ' + str(weightvalue) + ' at eigenvalue ' + str(eig[np.argmax(weight)]))
        weight = np.delete(weight, idx)
        eig = np.delete(eig, idx)
        idx = np.argmax(weight)
        weightvalue = weight[np.argmax(weight)]
        print('degeneracy value = ' + str(weightvalue) + ' at eigenvalue ' + str(eig[np.argmax(weight)]))
        weight = np.delete(weight, idx)
        eig = np.delete(eig, idx)
        weightvalue = weight[np.argmax(weight)]
        print('degeneracy value = ' + str(weightvalue) + ' at eigenvalue ' + str(eig[np.argmax(weight)]))
        print('degeneracy of largest eigenvalue = ' + str(weight[np.argmax(eig)]) + ' value = ' + str(eig[np.argmax(eig)]))
        print('degeneracy of smallest eigenvalue = ' + str(weight[np.argmin(eig)]) + ' value = ' + str(eig[np.argmin(eig)]))
