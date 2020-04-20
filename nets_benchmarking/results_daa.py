from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
from scipy import stats
warnings.filterwarnings('ignore') # suppresses warnings that arise because original code uses ...
# ... deprecated tf version after first execution of cell 

def create_data(version=2, normalscores=True):
    archs = np.array([[1,0],
                  [2,2],
                  [0,1]])
    arch_target = np.array([[1],[2],[0]])
    
    def generate_data (archs, arch_target, n_points, noise=0.1):
        if version==1:
            k = len(archs)
            X,Y = archs.T 
            rand = np.random.uniform (0,1,[k,n_points])
            rand = (rand/np.sum(rand,axis=0)).T
            joined = np.concatenate([archs, arch_target], axis=1)
            data = np.matmul(rand,joined)
            data = data + np.random.normal(0,noise,size=data.shape)
            feat, target = data[:,:-1], data[:,-1]
            return feat, target
        if version==2:
            k = len(archs)
            X,Y = archs.T 
            rand = np.random.uniform (0,1,[k,n_points])
            rand = (rand/np.sum(rand,axis=0)).T
            data = np.matmul(rand,archs)
            data = data + np.random.normal(0,noise,size=data.shape)
            Y0=np.linalg.norm(data-archs[0],axis=1)
            Y1=np.linalg.norm(data-archs[1],axis=1)
            Y2=np.linalg.norm(data-archs[2],axis=1)
            Y0=Y0/np.max(Y0)
            Y1=Y1/np.max(Y1)
            Y2=Y2/np.max(Y2)
            Y=np.concatenate((np.expand_dims(Y0,axis=1),np.expand_dims(Y1,axis=1),np.expand_dims(Y2,axis=1)),axis=1)
            return data, Y
    
    def normal_scores(list_of_arrays):
    
        def cumuldistr(counts, avoid1=True):
            """avoid1==True is needed to ensure that max of given array 
            is not encoded as inf by subsequent function gausspos"""
            if avoid1==True: counts = np.append(counts,[1],axis=0)
            counts = counts/np.sum(counts)
            for i in range(len(counts)):
                counts[i] = np.sum(counts[max(i-1,0):i+1])
            if avoid1==True: counts = counts[:-1]
            return counts
        
        def gausspos(cumuldistr):
            return stats.norm.ppf(cumuldistr)
        
        def normal_scores_per_col(col):
            col = col.astype("float")
            unique, counts = np.unique(col, return_counts=True)
            map2gauss = dict(zip(unique,gausspos(cumuldistr(counts))))
            for j in range(len(col)):
                col[j] = map2gauss[col[j]]
            return col
        
        for ind in range(len(list_of_arrays)):
            arr = list_of_arrays[ind]
            if arr.ndim==1: arr = normal_scores_per_col(arr)
            if arr.ndim==2: 
                concat_lst = [np.array([normal_scores_per_col(arr[:,i])]).T for i in range(arr.shape[1])]
                arr = np.concatenate(tuple(concat_lst),axis=1)
            if arr.ndim>2: raise Exception('All arrays in list  must have ndim of 1 or 2')
            list_of_arrays[ind] = arr
            
        return list_of_arrays
    
    x_train_feat, x_train_targets = generate_data (archs,arch_target,100000,noise=0.01)
    x_test_feat, x_test_targets = generate_data (archs,arch_target,1000,noise=0.01)
    
    if normalscores==True:
        [x_train_feat, x_train_targets,x_test_feat, x_test_targets] = normal_scores(
                                        [x_train_feat, x_train_targets,x_test_feat, x_test_targets])
    
    datadict = {'train_feat': x_train_feat, 'train_targets': x_train_targets, 
                'test_feat': x_test_feat, 'test_targets': x_test_targets}
    return datadict

def collect_results(data,
                    into = 'results', 
                    version = 'original', # 'luigi' 'milena'
                    at_loss_factor=8.0, 
                    target_loss_factor=8.0,
                    recon_loss_factor=4.0,
                    kl_loss_factor=4.0,
                    anneal = 0): 
    """Runs orginal daa code for version = 'original' or Milenas version for version = 'milena' and stores them into into"""
    if version=='luigi':
        from nets_benchmarking import  daa_luigi
        res = daa_luigi.build_network()(data, at_loss_factor, target_loss_factor,recon_loss_factor,kl_loss_factor, anneal)
    else:
        from nets_benchmarking import  daa
        res = daa.execute(data,version,at_loss_factor,target_loss_factor,recon_loss_factor,kl_loss_factor, anneal)
    # load and dump pickled results to enable comparison of luigis and other versions:
    try:
        with open(into, 'rb') as pickled_results:
            results = pickle.load(pickled_results)
    except:
        results = OrderedDict()
        
    def newkey(newdict,collectdict):
        key, value = list(newdict.items())[0]
        key_len = len(key)
        
        if key in list(collectdict.keys()):
            key = key + (1,)
            while key in list(collectdict.keys()):
                key = key[:key_len] + (key[key_len] + 1,)
        
        collectdict.update({key: value})

    newkey(res,results)
    
    with open(into,'wb') as file:
        pickle.dump(results,file)
    
def works(func, *args, **kwargs):
        try: 
            func(*args,**kwargs)
            return True
        except:
            return False
        
def unpack(seq,newtype=None):
    '''If type==None, generators will return generators, other outer container types will be preserved.
        -> Example: (1,[2],3) will return (1,2,3), [1,tuple(3,4)] will return [1,3,4] etc...
        type can also be set to any container type as list, tuple.'''
    import types
    def helpfunc(seq,helplst=[]):
        for i in seq:
            if works(len,i): helpfunc(i,helplst)
            else: helplst += [i]
        newseq = (j for j in helplst)
        return newseq
    
    newseq = helpfunc(seq,[])
    if newtype is not None:
        newseq = newtype(newseq)
    else:
        if not isinstance(seq,types.GeneratorType): 
            newseq = type(seq)(newseq)
    return newseq    

def rescale(a):
    """for instance, colors in rgb or list format:
    a = np.array([[9,4,3],[3,1,1],[15,0,2]])
    b = [2,10,4]"""
    c = np.array(a)
    if c.ndim==1: c = c.reshape((c.shape[-1],-1))
    cmin,cmax = np.array(c.min(axis=0)).reshape((-1,c.shape[1])), np.array(c.max(axis=0)).reshape((-1,c.shape[1]))
    c = (c-cmin)/cmax
    if c.ndim==1: c = c.reshape((c.shape[-1],-1))
    return c

def plot_results(pickled_results_path):
    """ Plots results comparingly, with lastly added results first."""
    with open(pickled_results_path, 'rb') as pickled_results:
        results = pickle.load(pickled_results)
        
    n_res = len(results.keys()) if len(results.keys())>1 else 2 # avoids weird error for subplots with 1 row
    height_fig = n_res*3
    fig, axs = plt.subplots(n_res, 3,figsize=(15,height_fig))

    modelpara_lst = results.keys() if len(results.keys())==1 else list(results.keys())[::-1]
    for modelpara_ind, modelpara in enumerate(modelpara_lst):
        df = results[modelpara]
        version = modelpara[0]
        for space_ind, space in enumerate(df.index):
            def normalize_colors():
                c=df.loc[space,'target_color']
                spaces = df.index if version!='luigi' else [list(df.index)[i] for i in [0,2]]
                tarvals = unpack([np.array(df.loc[space,'target_color']) for space in spaces])
                tarvals = [float(i) for i in tarvals]
                tarmin = min(tarvals)
                tarmax = max(tarvals)
                c = (c-tarmin)/(tarmax-tarmin) if np.array(c).ndim > 1 else [(i-tarmin)/tarmax for i in c]
                return np.array(c)
            
            c = normalize_colors() #if space!='latent space' else None
            
            axs[modelpara_ind,space_ind].scatter(df.loc[space,'dim1'],df.loc[space,'dim2'], c=c)
            
            if modelpara_ind==0: axs[modelpara_ind,space_ind].set_title(space, size=15) #fontdict=fontdict)
            if space_ind==0: axs[modelpara_ind,space_ind].set_ylabel(str(modelpara), fontsize=10) #(ylabel=str(modelpara))
        
    plt.show()