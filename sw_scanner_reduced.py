import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pickle
import tqdm
# import pandas as pd
from sw_scanner_lib import round_up_to_minute, round_down_to_minute,f
from gc import collect
import sys
import time
from pathlib import Path


def SolarWindScanner(
    Btot,
    Dist_au,
    settings = None,
    verbose = False,
    Ncores = 8,
    use_pandas = True,
    chunksize = 100,
    mininterval = 10,
    collect_garbage = True,
    collect_cadence = 10,
    batchsize = 1000000,
    savpath = None,
    savname = None,
    return_scans = False,
    save_dataframe = True,
    save_pickle = False,
    loop_dir = 'y',
):

    """
    Solar Wind Scanner
    Input:
        Btot: numpy array of float
        Dist_au: numpy array of float
        settings: dictionary
            wins: numpy array of np.timedelta64
            step: np.timedelta64
            normality_mode: str
                divergence: calculate the Jensen-Shannon Distance
                find_nan: calculate the nan ratio
            
    """

    if use_pandas:
        import pandas as pd

    if verbose:
        print("\n")
        print("-------------------------------------------")
        print("        Solar Wind Scanner Parallel")
        print("-------------------------------------------")
        print("\n")
        print("---- Settings ----\n")
        for k,v in settings.items():
            if k == 'wins':
                if use_pandas:
                    print("%s %s --> %s"%(k, pd.Timedelta(np.min(v)), pd.Timedelta(np.max(v))))
                else:
                    print("%s %s --> %s"%(k, np.min(v), np.max(v)))
            else:
                if use_pandas:
                    if type(v) is np.timedelta64:
                        print(k, pd.Timedelta(v))
                    else:
                        print(k, v)
                else:
                    print(k,v)

        print("\n---- End of settings ----\n\n")

    wins = settings['wins']
    step0 = settings['step']
    xgrid = settings['xgrid']
    xinds = settings['xinds']

    steps = np.zeros_like(wins)

    step_ratio_set = 1/0.01

    step_ratio_0 = int(wins[0]/step0)

    tstart_list = []
    tend_list = []
    for i1, win in enumerate(wins):
        step_ratio = (win/step0)
        if step_ratio < step_ratio_set:
            step_factor = 1
        else:
            step_factor = np.floor(step_ratio/step_ratio_set)

        steps[i1] = step0 * step_factor

        print(steps[i1], xgrid[0], xgrid[-1], int((xgrid[-1]-xgrid[0])/steps[i1]), print(type(xgrid[-1])))

        # calculate tstart_list and tend_list
        tstart_list_temp = np.arange(xgrid[0], xgrid[-1], np.timedelta64(steps[i1],'ns'))
        tend_list_temp = tstart_list_temp + win
        if i1 == 0 :
            tstart_list = tstart_list_temp
            tend_list = tend_list_temp
        else:
            tstart_list = np.append(tstart_list, tstart_list_temp)
            tend_list = np.append(tend_list, tend_list_temp)

    # randomize tstart_list and tend_list with the same indices
    inds = np.arange(len(tstart_list))
    np.random.shuffle(inds)
    tstart_list = tstart_list[inds]
    tend_list = tend_list[inds]


    N = len(xgrid) * len(wins)
    N1 = len(tstart_list)



    # print the input value range and Btot index range
    print('xgrid: %s - %s' %(xgrid[0], xgrid[-1]))

    print("\nTotal Pixels= %d\n" %(N))
    print("Total Iterations= %d\n" %(N1))

    alloc_input = {
        'Btot': Btot,
        'Dist_au': Dist_au,
        'settings': settings,
        'tstart_list': tstart_list,
        'tend_list': tend_list,
    }

    total_size = 0
    for key, value in alloc_input.items():
        size_in_bytes = sys.getsizeof(value)
        size_in_mb = size_in_bytes / (1024 * 1024)
        total_size += size_in_mb
        print("Size of %s: %.4f MB" %(key, size_in_mb))

    print(">>> Total size: %.4f MB <<<\n" %(total_size))

    settings['collect_garbage'] = collect_garbage
    settings['collect_cadence'] = collect_cadence
    settings['loop_dir'] = loop_dir
    print("\nCurrent Chunk size: %d" % chunksize)
    print("Garbage collection: %s, cadance: %d" % (collect_garbage, collect_cadence))
    print("Number of cores: %d" % Ncores)
    print("Loop dir: %s" % loop_dir)
    print("Savpath: ", savpath)
    print("Savname: ", savname)
    print("Return scans: ", return_scans)
    print("\n>>>>>>> Start Scanner Parallel >>>>>>>\n")

    # return 0

    if return_scans:
        scans = []
    else:
        scans = None


    t0 = time.time()

    # check the existing pickle files:
    if save_pickle:
        final_name = Path(savpath).joinpath(savname % (0))
        if Path(final_name).is_file():
            print("File already exists: %s\n" % final_name)
            # continue
    
    # check the existing dataframe files:
    if save_dataframe:
        final_name = Path(savpath).joinpath('df_'+savname % (0))
        if Path(final_name).is_file():
            print("File already exists: %s\n" % final_name)
            # continue
    
    # ===== main loop ===== #
    scanstemp = []
    with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
        for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(0, N1), chunksize), total=N1, mininterval=mininterval):
            scanstemp.append(scan)

    # save the file
    if save_pickle:
        final_name = Path(savpath).joinpath(savname % (0))
        with open(final_name, 'wb') as f:
            pickle.dump(scanstemp, f)
        print("Saved at: %s\n" % final_name)

    # turn scans into dataframe
    if save_dataframe:
        import pandas as pd
        dfpath = Path(savpath).joinpath('df_'+savname % (0))
        scansdf = pd.DataFrame(scanstemp)
        pd.to_pickle(scansdf, dfpath)
        print("Saved at: %s\n" % dfpath)

    if return_scans:
        scans += scanstemp

    return scans


def InitParallelAllocation(alloc_input):
    global Btot, Dist_au, settings, tstart_list, tend_list
    Btot = alloc_input['Btot']
    Dist_au = alloc_input['Dist_au']
    settings = alloc_input['settings']
    tstart_list = alloc_input['tstart_list']
    tend_list = alloc_input['tend_list']



def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global Btot, Dist_au, settings, tstart_list, tend_list

    n_sigma = settings['n_sigma']
    normality_mode = settings['normality_mode']
    step = settings['step']
    xinds = settings['xinds']
    xgrid = settings['xgrid']
    wins = settings['wins']
    capsize = settings['capsize']


    collect_garbage = settings['collect_garbage']
    collect_cadence = int(settings['collect_cadence'])


    tstart= tstart_list[i1]
    tend = tend_list[i1]
    win = tend - tstart

    try:
        id0 = xinds[int(np.where(xgrid == tstart)[0][0])]
        id1 = xinds[int(np.where(xgrid == tend)[0][0])]

        skip_size = int(np.floor((id1-id0)/capsize))
        if skip_size == 0:
            skip_size = 1

        btot = Btot[id0:id1:skip_size]

        nan_infos = {
            'ids': {'id0': id0, 'id1': id1, 'skip_size': skip_size, 'len': len(btot)}
        }
        nan_infos['flag'] = 1
    except:
        btot = None
        nan_infos = {
            'ids': {'id0': np.nan, 'id1': np.nan, 'skip_size': np.nan, 'len': np.nan}
        }
        nan_infos['flag'] = 0

    if normality_mode == 'fit':
        # r = np.copy(Dist_au[id0:id1:skip_size])
        divergence = settings['divergence']

        try:
            r = Dist_au[id0:id1:skip_size]

            # find the rescaling scale with r
            rfit = curve_fit(f, np.log10(r), np.log10(btot))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)

            # rescale x
            x = btot1
            x = (x-np.mean(x))/np.std(x)

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            outside_count = np.sum(x < bins[0]) + np.sum(x > bins[-1])


            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'js': js_div,
                'r_ratio': r_ratio,
                'fit_results': rfit,
                'fit_index': -rfit[0][0],
                'fit_pcovdiag': np.sqrt(np.diag(rfit[1]))[0],
                'outside_count': outside_count,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean_unscaled': np.mean(btot),
                'std_unscaled': np.std(btot),
                'skew_unscaled': stats.skew(btot),
                'kurt_unscaled': stats.kurtosis(btot),
                'mean': np.mean(btot1),
                'std': np.std(btot1),
                'skew': stats.skew(btot1),
                'kurt': stats.kurtosis(btot1)
            }


        except:
            # raise ValueError("fuck")
            rfit = (
                np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
            )
            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'js': np.nan,
                'r_ratio': np.nan,
                'fit_results': rfit,
                'fit_index': np.nan,
                'fit_pcovdiag': np.nan,
                'outside_count': np.nan,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean_unscaled': np.nan,
                'std_unscaled': np.nan,
                'skew_unscaled': np.nan,
                'kurt_unscaled': np.nan,
                'mean': np.nan,
                'std': np.nan,
                'skew': np.nan,
                'kurt': np.nan
            }


    elif normality_mode == 'no_fit':

        divergence = settings['divergence']

        try:

            x = btot

            # rescale x
            x = (x-np.mean(x))/np.std(x)

            # calculate pdf of x
            nbins = divergence['js']['nbins']
            bins = np.linspace(-n_sigma, n_sigma, nbins)
            hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
            outside_count = np.sum(x < bins[0]) + np.sum(x > bins[-1])


            # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
            bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
            pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

            js_div = jensenshannon(hist_data, pdf_gaussian)

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'js': js_div,
                'outside_count': outside_count,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean': np.mean(btot),
                'std': np.std(btot),
                'skew': stats.skew(btot),
                'kurt': stats.kurtosis(btot)
            }


        except:
            # raise ValueError("fuck")
            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'js': np.nan,
                'outside_count': np.nan,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean': np.nan,
                'std': np.nan,
                'skew': np.nan,
                'kurt': np.nan
            }

    elif normality_mode == 'hist_moments_no_fit':


        try:

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean': np.mean(btot),
                'std': np.std(btot),
                'skew': stats.skew(btot),
                'kurt': stats.kurtosis(btot)
            }


        except:
            # raise ValueError("fuck")
            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'tmid': tstart + win/2,
                'len': nan_infos['ids']['len'],
                'nan_flag': nan_infos['flag'],
                'id0': nan_infos['ids']['id0'],
                'id1': nan_infos['ids']['id1'],
                'skip_size': nan_infos['ids']['skip_size'],
                'mean': np.nan,
                'std': np.nan,
                'skew': np.nan,
                'kurt': np.nan
            }

    elif normality_mode == 'all_inclusive':
        r = Dist_au[id0:id1:skip_size]
        divergence = settings['divergence']
        nbins = divergence['js']['nbins']
        n_sigma = settings['n_sigma']

        try:

            # calculate the distance
            distances = {}
            outside_counts = {}

            x = btot

            # unscaled js
            js_div, outside_count = js_distance(x, n_sigma, nbins)
            distances['js'] = js_div
            outside_counts['js'] = outside_count
            
            # unscaled log10 js
            js_div, outside_count = js_distance(np.log10(x), n_sigma, nbins)
            distances['js_log10'] = js_div
            outside_counts['js_log10'] = outside_count

            # find the rescaling scale with r
            rfit = curve_fit(f, np.log10(r), np.log10(btot))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)

            # dist ratio
            r_ratio = np.max(r)/np.min(r)
            
            x = btot1

            # normal js
            js_div, outside_count = js_distance(x, n_sigma, nbins)
            distances['js_scaled'] = js_div
            outside_counts['js_scaled'] = outside_count

            # log10 js
            js_div, outside_count = js_distance(np.log10(x), n_sigma, nbins)
            distances['js_scaled_log10'] = js_div
            outside_counts['js_scaled_log10'] = outside_count


            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'r_ratio': r_ratio,
                'nan_infos': nan_infos,
                'fit_results': rfit,
                'outside_counts': outside_counts
            }


        except:
            # raise ValueError("fuck")
            rfit = (
                np.array([np.nan,np.nan]), np.array([[np.nan,np.nan],[np.nan,np.nan]])
            )
            distances = {
                'js':np.nan,
                'js_scaled':np.nan,
                'js_scaled_log10':np.nan,
                'js_log10':np.nan,
            }
            outside_counts = {
                'js':np.nan,
                'js_scaled':np.nan,
                'js_scaled_log10':np.nan,
                'js_log10':np.nan,
            }

            scan = {
                't0': tstart,
                't1': tend,
                'win': win,
                'distances': distances,
                'nan_infos': nan_infos,
                'fit_results': rfit,
                'outside_counts': outside_counts
            }



    elif normality_mode == 'find_nan':

        scan = {
            't0': tstart,
            't1': tend,
            'win': win,
            'nan_infos': nan_infos
        }


    else:
        raise ValueError("Wrong mode: %s" %(normality_mode))

    if collect_garbage:
        if i1 % collect_cadence == 0:
            collect()

    return scan





def js_distance(x, n_sigma, nbins):

    # rescale x
    x = (x-np.mean(x))/np.std(x)

    # calculate pdf of x
    bins = np.linspace(-n_sigma, n_sigma, nbins)
    hist_data, bin_edges_data = np.histogram(x, bins=bins, density=True)
    outside_count = np.sum(x < bins[0]) + np.sum(x > bins[-1])

    # Compute the PDF of the Gaussian distribution at the mid-points of the histogram bins
    bin_midpoints = bin_edges_data[:-1] + np.diff(bin_edges_data) / 2
    pdf_gaussian = stats.norm.pdf(bin_midpoints, 0, 1)

    js_div = jensenshannon(hist_data, pdf_gaussian)

    return js_div, outside_count