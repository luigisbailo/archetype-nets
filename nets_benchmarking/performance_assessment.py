import itertools as itt
from os import path
import pandas as pd
import numpy as np 
from scipy.stats import multivariate_normal
import os 
import pickle 
from collections import OrderedDict

def get_zfixed ( dim_latent_space ):
            
    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2
  
        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
                    
    return z_fixed

def points_in_simplex(zfixed,fineness):
    """Calculates positions of equally distributed points in simplex, 
    whose distance to their neighbors decreases with fineness."""
    dim_latent_space = zfixed.shape[1]
    
    res_path = "results_collections"
    filename = 'points_in_triangle({},{})'.format(dim_latent_space,fineness)
    
    if path.exists("{}/{}".format(res_path,filename)):
        points = np.array(pd.read_csv("{}/{}".format(res_path,filename)))
    
    else:
        
        def weights(dim_latent_space=dim_latent_space,fineness=fineness):
            x = np.expand_dims(np.linspace(0,1,fineness+1),axis=1)
            A = np.array([[0]*(dim_latent_space+1)])
            for i in list(itt.product(*[x]*(dim_latent_space+1)))[1:]:
                if np.sum(np.array(i))<=1:
                    A = np.concatenate((A,np.array(i).T),axis=0)
            return A

        points = np.matmul(weights(),zfixed)
        pd.DataFrame(points).to_csv(filename, index=False)
    
    return points

def hellingerdistance(mus,sigmas,points_in_simplex):
    """Calculates metric between [0,1] of "distance" between two distributions (0->same distributions)
      mus: np.array: rows=dta pts, cols=coords
      sigmas= np.array: rows=dta pts, only 1 column
      points_in_simplex: calculated with same-named-function"""
      
    def tdistrsummed_in_simplex(mus=mus,sigmas=sigmas,points_in_simplex=points_in_simplex):
        print('here normally sigma error!')
        for i in range(mus.shape[0]):
            sigmas = np.abs(sigmas)
            try:
                multigauss = multivariate_normal(mus[i,:], np.diagflat(sigmas[i]))
            except:
                print(mus[i,:],sigmas[i])
                multigauss = multivariate_normal(mus[i,:], np.diagflat(sigmas[i]))
            tdistr = multigauss.pdf(points_in_simplex)
            if i==0: tdistrsummed = tdistr
            if i>0:  tdistrsummed = tdistrsummed+tdistr
        tdistrsummed = tdistrsummed/np.sum(tdistrsummed) #integral over simplex is 1
        return tdistrsummed
    
    def uniform_in_simplex(points_in_simplex):
        n_points = points_in_simplex.shape[0]
        uniform = np.ones(n_points)/n_points # integral over simplex is one
        return uniform
    
    def gauss_in_simplex(points_in_simplex):
        dim_latent_space = points_in_simplex.shape[1]
        gauss = multivariate_normal(np.zeros(dim_latent_space,dtype=float), np.eye(dim_latent_space,dtype=float))
        simplexgauss = gauss.pdf(points_in_simplex) 
        simplexgauss = simplexgauss/np.sum(simplexgauss) #renormalizes gaussian so that integral over simplex is 1
        return simplexgauss
    
    all_t = tdistrsummed_in_simplex(mus,sigmas)
    
    bc_gauss = np.sum((all_t*gauss_in_simplex(points_in_simplex))**0.5)
    bc_uniform = np.sum((all_t*uniform_in_simplex(points_in_simplex))**0.5)
    H_gauss = (1-bc_gauss)**0.5
    H_uniform = (1-bc_uniform)**0.5
    
    return H_gauss, H_uniform #metric, scales between 0 (same distributions) and 1 (no distr overlap), fulfills triangle inequality

def hyperpar_comparison(list_of_resultfiles,points_in_simplex,dumpinto='hyp', overwrite=False):
    """list_of_resultfiles must contain a collection of filenames; each filename indicates the hyperparameters 
    on which the collection of results contained in the file are based.
    The filename could exhibit the following form (as preset in daa_exe.ipynb):
    modelparams = [normalscores,
               version,at_loss_factor,target_loss_factor,
               recon_loss_factor,kl_loss_factor,anneal]

    res_filename = 'res_{}'.format(tuple(modelparams)) """
    res_path = "results_collections"
    if overwrite is True or (overwrite is False and not path.exists(dumpinto)):
        
        hyperpar_df = pd.DataFrame(columns=['MSE_feats','MSE_targets','lat_gauss','lat_uniform'])
        for filename in list_of_resultfiles:
            
            def load_results(filename):
                try: 
                    os.mkdir(res_path)
                except: 
                    pass
                try:
                    with open("{}/{}".format(res_path,filename), 'rb') as pickled_results:
                        results = pickle.load(pickled_results)
                except:
                    print("{} is not contained in the reults_collections folder".format(filename))
                
                return results
            
            key = filename #[4:]
            results = load_results(filename)
            
            MSEin,MSEtar,latgauss,latuni = [],[],[],[]
            for _,item in results.items():
                mus,sigmas = item.at['latent space','features']
                H_gauss, H_uniform = hellingerdistance(mus,sigmas,points_in_simplex)
            
                MSEin = MSEin + [np.mean((item.at['real space','features'] - item.at['reconstructed real space','features'])**2)]
                MSEtar = MSEtar + [np.mean((item.at['real space','target_color'] - item.at['reconstructed real space','target_color'])**2)]
                
                latgauss = latgauss + [H_gauss]
                latuni = latuni + [H_uniform]
                
            
            hyperpar_df.at[key,'MSE_feats'] = tuple([np.mean(MSEin),np.std(MSEin)])
            hyperpar_df.at[key,'MSE_targets'] = tuple([np.mean(MSEtar),np.std(MSEtar)]) 
            hyperpar_df.at[key,'lat_gauss'] = tuple([np.mean(latgauss),np.std(latgauss)])
            hyperpar_df.at[key,'lat_uniform'] = tuple([np.mean(latuni),np.std(latuni)])
            
        
        if path.exists(dumpinto):
            dumpinto = dumpinto + '1'
        
        with open("{}/{}".format(res_path,dumpinto), 'wb') as file:
            pickle.dump(hyperpar_df, file)
        
    else:
        with open("{}/{}".format(res_path,dumpinto), 'rb') as pickled_results:
            hyperpar_df = pickle.load(pickled_results)

    return hyperpar_df
        
        



        