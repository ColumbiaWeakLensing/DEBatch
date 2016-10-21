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

fiducial = batch.getModel("Om0.260_Ode0.740_w-1.000_wa0.000_si0.800")
variations = ( 

	map(lambda m:batch.getModel(m),["Om0.290_Ode0.710_w-1.000_wa0.000_si0.800","Om0.260_Ode0.740_w-0.800_wa0.000_si0.800","Om0.260_Ode0.740_w-1.000_wa0.000_si0.900"]),
	map(lambda m:batch.getModel(m),["Om0.230_Ode0.770_w-1.000_wa0.000_si0.800","Om0.260_Ode0.740_w-1.200_wa0.000_si0.800","Om0.260_Ode0.740_w-1.000_wa0.000_si0.700"])

)

plab = { "Om":r"$\Omega_m$", "w0":r"$w_0$", "wa":r"$w_a$", "si8":r"$\sigma_8$" }

###################################################################################################
###################################################################################################

def pbBias(cmd_args,feature_name="convergence_power_s0_nb100",kappa_models=("Born",),callback=None,variation_idx=(0,1),bootstrap_size=100,resample=1000,fontsize=22):
	
	#Initialize plot
	fig,ax = plt.subplots(len(variation_idx),3,figsize=(24,8*len(variation_idx)))
	ax = np.atleast_2d(ax)

	##################
	#Load in the data#
	##################

	#Observation
	bootstrap_mean = lambda e: e.values.mean(0)
	feature_ray = Ensemble.read(os.path.join(fiducial["c0"].getMapSet("kappa").home,feature_name+".npy"),callback_loader=callback).bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample)

	#Containers for cosmological model
	modelFeatures = dict()
	for mf in kappa_models:
		modelFeatures[mf] = dict()

	parameters = dict()

	for model in models:
		parameters[model.cosmo_id] = np.array([model.cosmology.Om0,model.cosmology.w0,model.cosmology.sigma8])
		for mf in kappa_models:
			modelFeatures[mf][model.cosmo_id] = Ensemble.read(os.path.join(model["c0"].getMapSet("kappa"+mf).home,feature_name+".npy"),callback_loader=callback)

	#Fit each model
	for mf in kappa_models:

		#Select correct 
		features = modelFeatures[mf]

		###############################
		#Compute the covariance matrix#
		###############################

		features_covariance = features[fiducial.cosmo_id].cov()

		################################################
		#Load in the feature to fit, bootstrap the mean#
		################################################
	
		feature_born = features[fiducial.cosmo_id].bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample)

		for nv,v in enumerate(variation_idx):

			###############################
			#Initialize the FisherAnalysis#
			###############################

			ftr = np.array([features[m.cosmo_id].values.mean(0) for m in [fiducial] + variations[v]])
			par = np.array([parameters[m.cosmo_id] for m in [fiducial] + variations[v]])
			fisher = FisherAnalysis.from_features(ftr,par,parameter_index=["Om","w0","si8"])

			#############
			####Fit######
			#############

			fitted_parameters_born = fisher.fit(feature_born,features_covariance)
			fitted_parameters_ray = fisher.fit(feature_ray,features_covariance)

			##########
			#Plotting#
			##########

			for n,p in enumerate(fisher.parameter_names):
				fitted_parameters_born[p].plot.hist(bins=50,ax=ax[nv,n],label=mf+"(Mock)",alpha=0.4)
				fitted_parameters_ray[p].plot.hist(bins=50,ax=ax[nv,n],label=mf+"(Observation)",alpha=0.4)
				
				ax[nv,n].set_xlabel(plab[p],fontsize=18)
				ax[nv,n].set_title("Fisher {0}".format(v))
				ax[nv,n].legend()
	
	#Save
	fig.savefig("bias_{0}.{1}".format(feature_name,cmd_args.type))

####################################################################################################################

def pbBiasPower(cmd_args,feature_name="convergence_power_s0_nb100"):
	pbBias(cmd_args,feature_name=feature_name)

def pbBiasPowerSN(cmd_args,feature_name="convergence_powerSN_s0_nb100"):
	pbBias(cmd_args,feature_name=feature_name)

def pbBiasMoments(cmd_args,feature_name="convergence_moments_s0_nb9"):
	pbBias(cmd_args,feature_name=feature_name,kappa_models=("Born","BornRT"))

def pbBiasMomentsSN(cmd_args,feature_name="convergence_momentsSN_s0_nb9"):
	pbBias(cmd_args,feature_name=feature_name)

def pbBiasMomentsSN45(cmd_args,feature_name="convergence_momentsSN45_s0_nb9"):
	pbBias(cmd_args,feature_name=feature_name)

def pbBiasSkew(cmd_args,feature_name="convergence_skew_s0_nb9"):
	callback = lambda f:np.load(f.replace("skew","moments"))[:,2:5]
	pbBias(cmd_args,feature_name=feature_name,callback=callback)

def pbBiasSkewSN(cmd_args,feature_name="convergence_skewSN_s0_nb9"):
	callback = lambda f:np.load(f.replace("skew","moments"))[:,2:5]
	pbBias(cmd_args,feature_name=feature_name,callback=callback)

def pbBiasKurt(cmd_args,feature_name="convergence_kurt_s0_nb9"):
	callback = lambda f:np.load(f.replace("kurt","moments"))[:,5:]
	pbBias(cmd_args,feature_name=feature_name,callback=callback)

def pbBiasKurtSN(cmd_args,feature_name="convergence_kurtSN_s0_nb9"):
	callback = lambda f:np.load(f.replace("kurt","moments"))[:,5:]
	pbBias(cmd_args,feature_name=feature_name,callback=callback)

####################################################################################################################
####################################################################################################################

def pdfPlot(cmd_args,features,figname,fontsize=22):

	#Initialize plot
	fig,ax = plt.subplots()

	#Load features and plot
	for f in features:
		model,mapset,name,bn = features[f]
		fname = os.path.join(model["c0"].getMapSet(mapset).home,name+".npy")
		samples = np.load(fname)[:,bn]
		ax.hist(samples,bins=50,alpha=0.4,label=f)

	#Legend 
	ax.legend()

	#Save
	fig.savefig(figname+"."+cmd_args.type)

def pdfMoments(cmd_args,kappa_models=("kappa","kappaBorn","kappaBornRT"),figname="pdfMoments",fontsize=22):

	#Moment labels
	moment_labels = (r"$\sigma_0^2$",r"$\sigma_1^2$",r"$S_0$",r"$S_1$",r"$S_2$",r"$K_0$",r"$K_1$",r"$K_2$",r"$K_3$")

	#Set up plot
	fig,axes = plt.subplots(3,3,figsize=(24,)*2)

	#Bootstrap mean to mimic LSST
	bootstrap_mean = lambda e: e.values.mean(0)

	#PDF for each kappa model
	for km in kappa_models:

		#Load samples
		fname = os.path.join(fiducial["c0"].getMapSet(km).home,"convergence_moments_s0_nb9.npy")
		samples = Ensemble.read(fname).bootstrap(bootstrap_mean,bootstrap_size=1000,resample=1000)

		#Plot each bin
		for bn,ax in enumerate(axes.flatten()):
			ax.hist(samples[:,bn],bins=50,alpha=0.4,label=km)

	#Axes labels
	for bn,ax in enumerate(axes.flatten()):
		ax.legend()
		ax.set_title(moment_labels[bn],fontsize=fontsize)

	#Save
	fig.tight_layout()
	fig.savefig(figname+".{0}".format(cmd_args.type))

####################################################################################################################

def pdfSkew(cmd_args):

	features = {
		r"$S_0({\rm noiseless})$" : (fiducial,"kappa","convergence_moments_s0_nb9",3),
		r"$S_0({\rm noiseless (Born)})$" : (fiducial,"kappaBorn","convergence_moments_s0_nb9",3),
		r"$S_0({\rm noiseless (RT)})$" : (fiducial,"kappaBornRT","convergence_moments_s0_nb9",3)
	}

	figname = "pdfSkew"

	pdfPlot(cmd_args,features=features,figname=figname)

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()

method["1"] = pbBiasPower
method["1b"] = pbBiasPowerSN

method["2"] = pbBiasMoments
method["2b"] = pbBiasMomentsSN
method["2b45"] = pbBiasMomentsSN45
method["2c"] = pbBiasSkew
method["2d"] = pbBiasSkewSN
method["2e"] = pbBiasKurt
method["2f"] = pbBiasKurtSN

method["3"] = pdfMoments

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()