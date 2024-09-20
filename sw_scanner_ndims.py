import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy import stats
from scipy.spatial.distance import jensenshannon
import pickle
import tqdm
# import pandas as pd
from sw_scanner_lib import f, js_distance
from gc import collect
import sys
import time
from pathlib import Path


def SolarWindScanner(
    Btot,
    Dist_au,
    Bvec,
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
        print("-------------------------------------------------")
        print("        Solar Wind Scanner Parallel Ndims")
        print("-------------------------------------------------")
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
    step = settings['step']
    xgrid = settings['xgrid']

    N = len(xgrid) * len(wins)


    # print the input value range and Btot index range
    print('xgrid: %s - %s' %(xgrid[0], xgrid[-1]))

    print("\nTotal intervals= %d\n" %(N))

    alloc_input = {
        'Btot': Btot,
        'Bvec': Bvec,
        'Dist_au': Dist_au,
        'settings': settings,
        # 'win_list': win_list,
        # 'nsteps_list': nsteps_list,
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
    print("Batch size: %d" % batchsize)
    print("Loop dir: %s" % loop_dir)
    print("Savpath: ", savpath)
    print("Savname: ", savname)
    print("Return scans: ", return_scans)
    print("\n>>>>>>> Start Scanner Parallel >>>>>>>\n")

    if return_scans:
        scans = []
    else:
        scans = None

    # create batch size
    # imap_unordered initialize RAM size based on the MAXIMUM value in range(), NOT the number of loops
    # breaking into batches would significantly reduce the RAM usage
    Nbatch = N//batchsize
    t0 = time.time()
    for i1 in range(Nbatch+1):
        r0 = i1*batchsize
        if i1 < Nbatch:
            r1 = (i1+1)*batchsize
        else:
            r1 = N

        alloc_input['starting_index'] = r0
        scanstemp = []

        print(">>Batch %d out of %d: %d - %d, tspent: %.2f min" %(i1+1, Nbatch+1, r0, r1, (time.time()-t0)/60))

        # check the existing pickle files:
        if save_pickle:
            final_name = Path(savpath).joinpath(savname % (i1))
            if Path(final_name).is_file():
                print("File already exists: %s\n" % final_name)
                continue
        
        # check the existing dataframe files:
        if save_dataframe:
            final_name = Path(savpath).joinpath('df_'+savname % (i1))
            if Path(final_name).is_file():
                print("File already exists: %s\n" % final_name)
                continue
        
        # ===== main loop ===== #
        with Pool(Ncores, initializer=InitParallelAllocation, initargs=(alloc_input,)) as p:
            for scan in tqdm.tqdm(p.imap_unordered(SolarWindScannerInnerLoopParallel, range(0, r1-r0), chunksize), total=r1-r0, mininterval=mininterval):
                scanstemp.append(scan)

        # save the file
        if save_pickle:
            final_name = Path(savpath).joinpath(savname % (i1))
            with open(final_name, 'wb') as f:
                pickle.dump(scanstemp, f)
            print("Saved at: %s\n" % final_name)

        # turn scans into dataframe
        if save_dataframe:
            import pandas as pd
            dfpath = Path(savpath).joinpath('df_'+savname % (i1))
            scansdf = pd.DataFrame(scanstemp)
            pd.to_pickle(scansdf, dfpath)
            print("Saved at: %s\n" % dfpath)

        if return_scans:
            scans += scanstemp

    return scans


def InitParallelAllocation(alloc_input):
    global Btot, Bvec, Dist_au, settings, starting_index
    Btot = alloc_input['Btot']
    Bvec = alloc_input['Bvec']
    Dist_au = alloc_input['Dist_au']
    settings = alloc_input['settings']
    # win_list = alloc_input['win_list']
    # nsteps_list = alloc_input['nsteps_list']
    starting_index = alloc_input['starting_index']



def SolarWindScannerInnerLoopParallel(i1):
    # access global variables
    global Btot, Bvec, Dist_au, settings, starting_index

    i1 = starting_index + i1

    n_sigma = settings['n_sigma']
    normality_mode = settings['normality_mode']
    step = settings['step']
    xinds = settings['xinds']
    xgrid = settings['xgrid']
    wins = settings['wins']
    capsize = settings['capsize']
    loop_dir = settings['loop_dir']


    collect_garbage = settings['collect_garbage']
    collect_cadence = int(settings['collect_cadence'])

    # loop through x
    if loop_dir == 'x':
        win = wins[i1//len(xgrid)]
        nsteps = i1 % len(xgrid)
    # loop through wins
    # this is more efficient compared to loop through x
    # because this minimizes the moving of pointers in RAM, would be 2~3x faster
    elif loop_dir == 'y':
        nsteps = i1//len(wins)
        win = wins[i1%len(wins)]
    else:
        raise NotImplementedError

    tstart0 = xgrid[0]
    tstart= tstart0 + nsteps * step
    tend = tstart + win

    try:
        id0 = xinds[int(np.where(xgrid == tstart)[0][0])]
        id1 = xinds[int(np.where(xgrid == tend)[0][0])]

        skip_size = int(np.floor((id1-id0)/capsize))
        if skip_size == 0:
            skip_size = 1

        btot = Btot[id0:id1:skip_size]
        bvec = Bvec[id0:id1:skip_size,:]

        nan_infos = {
            'ids': {'id0': id0, 'id1': id1, 'skip_size': skip_size, 'len': len(btot)}
        }
        nan_infos['flag'] = 1
    except:
        btot = None
        bvec = None
        nan_infos = {
            'ids': {'id0': np.nan, 'id1': np.nan, 'skip_size': np.nan, 'len': np.nan}
        }
        nan_infos['flag'] = 0

    if normality_mode == 'fit':
        divergence = settings['divergence']

        try:
            r = Dist_au[id0:id1:skip_size]

            # find the rescaling scale with r
            rfit = curve_fit(f, np.log10(r), np.log10(btot))
            scale = -rfit[0][0]

            # normalize btot with r
            btot1 = btot * ((r/r[0])**scale)
            bvec1 = bvec * np.reshape(((r/r[0])**scale), (len(r),1))

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
                'btot_mean_unscaled': np.mean(btot),
                'btot_std_unscaled': np.std(btot),
                'btot_skew_unscaled': stats.skew(btot),
                'btot_kurt_unscaled': stats.kurtosis(btot),
                'btot_mean': np.mean(btot1),
                'btot_std': np.std(btot1),
                'btot_skew': stats.skew(btot1),
                'btot_kurt': stats.kurtosis(btot1),
                'br_mean_unscaled': np.mean(bvec[:,0]),
                'br_std_unscaled': np.std(bvec[:,0]),
                'br_skew_unscaled': stats.skew(bvec[:,0]),
                'br_kurt_unscaled': stats.kurtosis(bvec[:,0]),
                'br_mean': np.mean(bvec1[:,0]),
                'br_std': np.std(bvec1[:,0]),
                'br_skew': stats.skew(bvec1[:,0]),
                'br_kurt': stats.kurtosis(bvec1[:,0]),
                'bt_mean_unscaled': np.mean(bvec[:,1]),
                'bt_std_unscaled': np.std(bvec[:,1]),
                'bt_skew_unscaled': stats.skew(bvec[:,1]),
                'bt_kurt_unscaled': stats.kurtosis(bvec[:,1]),
                'bt_mean': np.mean(bvec1[:,1]),
                'bt_std': np.std(bvec1[:,1]),
                'bt_skew': stats.skew(bvec1[:,1]),
                'bt_kurt': stats.kurtosis(bvec1[:,1]),
                'bn_mean_unscaled': np.mean(bvec[:,2]),
                'bn_std_unscaled': np.std(bvec[:,2]),
                'bn_skew_unscaled': stats.skew(bvec[:,2]),
                'bn_kurt_unscaled': stats.kurtosis(bvec[:,2]),
                'bn_mean': np.mean(bvec1[:,2]),
                'bn_std': np.std(bvec1[:,2]),
                'bn_skew': stats.skew(bvec1[:,2]),
                'bn_kurt': stats.kurtosis(bvec1[:,2]),
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
                'btot_mean_unscaled': np.nan,
                'btot_std_unscaled': np.nan,
                'btot_skew_unscaled': np.nan,
                'btot_kurt_unscaled': np.nan,
                'btot_mean': np.nan,
                'btot_std': np.nan,
                'btot_skew': np.nan,
                'btot_kurt': np.nan,
                'br_mean_unscaled': np.nan,
                'br_std_unscaled': np.nan,
                'br_skew_unscaled': np.nan,
                'br_kurt_unscaled': np.nan,
                'br_mean': np.nan,
                'br_std': np.nan,
                'br_skew': np.nan,
                'br_kurt': np.nan,
                'bt_mean_unscaled': np.nan,
                'bt_std_unscaled': np.nan,
                'bt_skew_unscaled': np.nan,
                'bt_kurt_unscaled': np.nan,
                'bt_mean': np.nan,
                'bt_std': np.nan,
                'bt_skew': np.nan,
                'bt_kurt': np.nan,
                'bn_mean_unscaled': np.nan,
                'bn_std_unscaled': np.nan,
                'bn_skew_unscaled': np.nan,
                'bn_kurt_unscaled': np.nan,
                'bn_mean': np.nan,
                'bn_std': np.nan,
                'bn_skew': np.nan,
                'bn_kurt': np.nan,
            }



    else:
        raise ValueError("Wrong mode: %s" %(normality_mode))

    if collect_garbage:
        if i1 % collect_cadence == 0:
            collect()

    return scan


