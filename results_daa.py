from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore') # suppresses warnings that arise because original code uses ...
# ... deprecated tf version after first execution of cell 

def create_data(version=2):
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
        
    x_train_feat, x_train_targets = generate_data (archs,arch_target,100000,noise=0.01)
    x_test_feat, x_test_targets = generate_data (archs,arch_target,1000,noise=0.01)
    datadict = {'train_feat': x_train_feat, 'train_targets': x_train_targets, 
                'test_feat': x_test_feat, 'test_targets': x_test_targets}
    return datadict

def collect_results(data,
                    into = 'results', 
                    version = 'original', # 'luigi' 'milena'
                    at_loss_factor=8.0, 
                    target_loss_factor=8.0,
                    recon_loss_factor=4.0,
                    kl_loss_factor=4.0): 
    """Runs orginal daa code for version = 'original' or Milenas version for version = 'milena' and stores them into into"""
    if version=='luigi':
        import daa_luigi
        res = daa_luigi.build_network()(data, at_loss_factor, target_loss_factor,recon_loss_factor,kl_loss_factor)
    else:
        import daa
        res = daa.execute(data,version,at_loss_factor,target_loss_factor,recon_loss_factor,kl_loss_factor)
    # load and dump pickled results to enable comparison of luigis and other versions:
    try:
        with open(into, 'rb') as pickled_results:
            results = pickle.load(pickled_results)
    except:
        results = OrderedDict()
        
    results.update(res) 
    
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
                c = (c-tarmin)/tarmax if np.array(c).ndim > 1 else [(i-tarmin)/tarmax for i in c]
                return c
            
            c = normalize_colors()
            
            print(modelpara,space,type(c),np.array(c).shape)
            axs[modelpara_ind,space_ind].scatter(df.loc[space,'dim1'],df.loc[space,'dim2'], c=c)
            
            if modelpara_ind==0: axs[modelpara_ind,space_ind].set_title(space, size=15) #fontdict=fontdict)
            if space_ind==0: axs[modelpara_ind,space_ind].set_ylabel(str(modelpara), fontsize=10) #(ylabel=str(modelpara))
        
    plt.show()