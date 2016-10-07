#!/usr/bin/env python
from __future__ import division

import sys,os,argparse
sys.modules["mpi4py"] = None

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import SimulationBatch

from lenstools.statistics.ensemble import Ensemble
from lenstools.statistics.constraints import FisherAnalysis

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

#########################################
#Global: SimulationBatch handle + models#
#########################################

batch = SimulationBatch.current()
models = batch.models

fiducial = models[4]
variations = ( [ models[n] for n in [6,1,2] ] , [ models[n] for n in [0,5,3] ] )

plab = { "Om":r"$\Omega_m$", "w0":r"$w_0$", "wa":r"$w_a$" }

###################################################################################################
###################################################################################################

def pbBias(cmd_args,feature_name="power_s0_nb98",variation_idx=0,fontsize=22):
	
	#Initialize plot
	fig,ax = plt.subplots(1,3,figsize=(24,8))

	##################
	#Load in the data#
	##################

	features = dict()
	parameters = dict()

	for model in models:
		features[model.cosmo_id] = Ensemble.read(os.path.join(model["c0"].getMapSet("kappaBorn").home,feature_name+".npy"))
		parameters[model.cosmo_id] = np.array([model.cosmology.Om0,model.cosmology.w0,model.cosmology.wa])

	###############################
	#Initialize the FisherAnalysis#
	###############################

	ftr = np.array([features[m.cosmo_id].values.mean(0) for m in [fiducial] + variations[variation_idx]])
	par = np.array([parameters[m.cosmo_id] for m in [fiducial] + variations[variation_idx]])
	fisher = FisherAnalysis.from_features(ftr,par,parameter_index=["Om","w0","wa"])

	###############################
	#Compute the covariance matrix#
	###############################

	features_covariance = features[fiducial.cosmo_id].cov()

	#############################################
	#Load in the feature to fit, perform the fit#
	#############################################

	feature_born = features[fiducial.cosmo_id]
	feature_ray = Ensemble.read(os.path.join(model["c0"].getMapSet("kappa").home,feature_name+".npy"))
	fitted_parameters_born = fisher.fit(feature_born,features_covariance)
	fitted_parameters_ray = fisher.fit(feature_ray,features_covariance)

	##########
	#Plotting#
	##########

	for n,p in enumerate(fisher.parameter_names):
		fitted_parameters_born[p].plot.hist(bins=50,ax=ax[n],label="Born",alpha=0.4)
		fitted_parameters_ray[p].plot.hist(bins=50,ax=ax[n],label="Ray",alpha=0.4)
		ax[n].set_xlabel(plab[p],fontsize=18)
		ax[n].legend()
	
	#Save
	fig.savefig("bias_{0}.{1}".format(feature_name,cmd_args.type))

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()
method["1"] = pbBias

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()