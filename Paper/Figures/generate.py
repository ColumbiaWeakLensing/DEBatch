#!/usr/bin/env python
from __future__ import division

import sys,os,argparse
from operator import add
from functools import reduce

sys.modules["mpi4py"] = None

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import astropy.units as u

sns.set(font_scale=2)

from lenstools.pipeline.simulation import SimulationBatch

from lenstools.image.convergence import ConvergenceMap
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

def convergenceVisualize(cmd_args,collection="c0",smooth=0.5*u.arcmin,fontsize=22):

	#Initialize plot
	fig,ax = plt.subplots(2,2,figsize=(16,16))

	#Load data
	cborn = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"born_z2.00_0001r.fits"))
	cray = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappa").home,"WLconv_z2.00_0001r.fits"))
	cll = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaLL").home,"postBorn2-ll_z2.00_0001r.fits"))
	cgp = ConvergenceMap.load(os.path.join(fiducial[collection].getMapSet("kappaGP").home,"postBorn2-gp_z2.00_0001r.fits"))

	#Smooth
	for c in (cborn,cray,cll,cgp):
		c.smooth(smooth,kind="gaussianFFT",inplace=True)

	#Plot
	cray.visualize(colorbar=True,fig=fig,ax=ax[0,0])
	(cray+cborn*-1).visualize(colorbar=True,fig=fig,ax=ax[0,1])
	cll.visualize(colorbar=True,fig=fig,ax=ax[1,0])
	cgp.visualize(colorbar=True,fig=fig,ax=ax[1,1])

	#Titles
	ax[0,0].set_title(r"$\kappa$",fontsize=fontsize)
	ax[0,1].set_title(r"$\kappa-\kappa_{\rm born}$",fontsize=fontsize)
	ax[1,0].set_title(r"$\kappa_{\rm ll}$",fontsize=fontsize)
	ax[1,1].set_title(r"$\kappa_{\rm geo}$",fontsize=fontsize)

	#Switch off grids
	for i in (0,1):
		for j in (0,1):
			ax[i,j].grid(b=False) 

	#Save
	fig.tight_layout()
	fig.savefig("csample."+cmd_args.type)

###################################################################################################
###################################################################################################

def powerResiduals(cmd_args,collection="c0",fontsize=22):

	#Initialize plot
	fig,ax = plt.subplots(1,2,figsize=(16,8))

	#Load data
	ell = np.load(os.path.join(batch.home,"ell_nb100.npy"))
	thetaG = 0.5*u.arcmin
	smooth = np.exp(-(ell*(thetaG.to(u.rad).value))**2)

	#Plot
	pFull = np.load(os.path.join(fiducial[collection].getMapSet("kappa").home,"convergence_power_s0_nb100.npy"))
	pBorn = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"convergence_power_s0_nb100.npy"))
	pLL = np.load(os.path.join(fiducial[collection].getMapSet("kappaLL").home,"convergence_power_s0_nb100.npy"))
	pLL_cross = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"cross_powerLL_s0_nb100.npy"))
	pGP = np.load(os.path.join(fiducial[collection].getMapSet("kappaGP").home,"convergence_power_s0_nb100.npy"))
	pGP_cross = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"cross_powerGP_s0_nb100.npy"))
	p_redshear_cross = np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"redshear_power_s0_nb100.npy"))

	#Plot
	ax[0].plot(ell,ell*(ell+1)*smooth*pBorn.mean(0)/(2.*np.pi),label=r"${\rm born}$")
	ax[0].plot(ell,ell*(ell+1)*smooth*pGP.mean(0)/(2.*np.pi),label=r"${\rm geodesic}$")
	ax[0].plot(ell,ell*(ell+1)*smooth*pLL.mean(0)/(2.*np.pi),label=r"${\rm lens-lens}$")
	ax[1].plot(ell,ell*(ell+1)*smooth*(np.abs(pFull.mean(0)-pBorn.mean(0)))/(2.0*np.pi),label=r"${\rm ray-born}$")
	ax[1].plot(ell,ell*(ell+1)*smooth*np.abs(pGP_cross.mean(0))/np.pi,label=r"$2{\rm born}\times{\rm geo}$")
	ax[1].plot(ell,ell*(ell+1)*smooth*np.abs(pLL_cross.mean(0))/np.pi,label=r"$2{\rm born}\times{\rm ll}$")
	ax[1].plot(ell,ell*(ell+1)*smooth*np.abs(p_redshear_cross.mean(0))/np.pi,label=r"$2\kappa_{\rm born}\times(\kappa\gamma)_{\rm born}$",color=sns.xkcd_rgb["dark grey"])
	ax[1].plot(ell,ell*(ell+1)*smooth*pFull.std(0)/(2.0*np.pi)/1000.,label=r"${\rm Cosmic}$ ${\rm variance (LSST)}$",color=sns.xkcd_rgb["pumpkin"])
	ax[1].plot(ell,ell*(ell+1)*smooth*np.abs(pFull.mean(0)-pBorn.mean(0)-2*pGP_cross.mean(0)-2*pLL_cross.mean(0))/(2.0*np.pi),label=r"${\rm ray-born}-2{\rm born}\times{\rm geo}-2{\rm born}\times{\rm ll}$",linestyle="--",color=sns.xkcd_rgb["denim blue"])

	#Shape noise
	pshape = ((0.15+0.035*2.0)**2)/((45*(u.arcmin**-2)).to(u.rad**-2).value)

	for n in (0,1):
		ax[n].plot(ell,ell*(ell+1)*pshape*smooth/(2.*np.pi),label=r"${\rm Shape}$ ${\rm noise,}$ $45{\rm gal/arcmin}^2$",color=sns.xkcd_rgb["dark grey"],linestyle="--")

	#Labels
	for n in (0,1):
		ax[n].set_xscale("log")
		ax[n].set_yscale("log")
		ax[n].set_xlabel(r"$\ell$",fontsize=fontsize)
		ax[n].legend(ncol=2,loc=3,bbox_to_anchor=(0., 1.02, 1., .102),prop={"size":15})
	
	ax[0].set_ylabel(r"$\ell(\ell+1)P^{\kappa\kappa}(\ell)/2\pi$",fontsize=fontsize)
	ax[1].set_ylabel(r"$\ell(\ell+1){\rm abs(residuals)/2\pi}$")

	#Save
	fig.tight_layout()
	fig.savefig("powerResiduals."+cmd_args.type)

####################################################################################################################
####################################################################################################################

def plotSmooth(cmd_args,lines,collection="c0",moment=2,smooth=(0.5,1.,2.,3.,5.,7.,10.),ylabel=None,fontsize=22):

	#Set up plot
	fig,ax = plt.subplots()

	#Load reference data
	reference = list()
	for s in smooth:
		reference.append(np.load(os.path.join(fiducial[collection].getMapSet("kappaBorn").home,"convergence_moments_s{0}_nb9.npy".format(int(s*100))))[:,moment].mean())
	reference = np.array(reference)

	#Plot each of the lines
	lk = lines.keys()
	lk.sort(key=lambda k:lines[k][-1])

	for l in lk:
		ms,feat,idx,subtract,color,linestyle,order = lines[l]
		data = list()

		for s in smooth:
			addends = [ np.load(os.path.join(fiducial[collection].getMapSet(ms).home,f.format(int(s*100))))[:,idx].mean() for f in feat ]
			data.append(reduce(add,addends))

		data = np.array(data)
		if subtract:
			data-=reference

		ax.plot(smooth,data/reference,color=sns.xkcd_rgb[color],linestyle=linestyle,label=l)


	#Labels
	ax.set_xlabel(r"$\theta_G({\rm arcmin})$",fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	lgd = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=2,mode="expand", borderaxespad=0.,prop={"size":15})

	#Save
	fig.tight_layout()
	fig.savefig("delta_m{0}.{1}".format(moment,cmd_args.type),bbox_extra_artists=(lgd,),bbox_inches="tight")

def plotSmoothSkew(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 2 

	#Lines to plot
	lines = {

	r"$\kappa^3_{\rm ray}-\kappa^3_{\rm born}$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$\kappa^3_{\rm born+geo}-\kappa^3_{\rm born}$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$\kappa^3_{\rm born+ll}-\kappa^3_{\rm born}$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$3\kappa^2_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_skewGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$3\kappa^2_{\rm born}\kappa_{\rm ll}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$3\kappa^2_{\rm born}(\kappa\gamma)_{\rm born}$" : ("kappaBorn",("redshear_skew_s{0}_nb1.npy",),0,False,"dark grey","-",6),
	r"$3\kappa^2_{\rm born}\kappa_{\rm ll}+3\kappa^2_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_skewLL_s{0}_nb1.npy","cross_skewGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^3\rangle/\langle\kappa^3\rangle$",fontsize=fontsize)

def plotSmoothKurt(cmd_args,collection="c0",smooth=(0.5,1.,2.,3.,5.,7.,10.),fontsize=22):

	moment = 5 

	#Lines to plot
	lines = {

	r"$\kappa^4_{\rm ray}-\kappa^4_{\rm born}$" : ("kappa",("convergence_moments_s{0}_nb9.npy",),moment,True,"denim blue","-",0),
	r"$\kappa^4_{\rm born+geo}-\kappa^4_{\rm born}$" : ("kappaB+GP",("convergence_moments_s{0}_nb9.npy",),moment,True,"medium green","-",1),
	r"$\kappa^4_{\rm born+ll}-\kappa^4_{\rm born}$" : ("kappaB+LL",("convergence_moments_s{0}_nb9.npy",),moment,True,"pale red","-",2),
	r"$4\kappa^3_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_kurtGP_s{0}_nb1.npy",),0,False,"medium green","--",4),
	r"$4\kappa^4_{\rm born}\kappa_{\rm ll}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy",),0,False,"pale red","--",5),
	r"$4\kappa^3_{\rm born}(\kappa\gamma)_{\rm born}$" : ("kappaBorn",("redshear_kurt_s{0}_nb1.npy",),0,False,"dark grey","-",6),
	r"$4\kappa^3_{\rm born}\kappa_{\rm ll}+4\kappa^3_{\rm born}\kappa_{\rm geo}$" : ("kappaBorn",("cross_kurtLL_s{0}_nb1.npy","cross_kurtGP_s{0}_nb1.npy"),0,False,"denim blue","--",3),

	}

	plotSmooth(cmd_args,lines,collection=collection,moment=moment,smooth=smooth,ylabel=r"$\langle\delta\kappa^4\rangle_c/\langle\kappa^4\rangle_c$",fontsize=fontsize)

####################################################################################################################
####################################################################################################################

def pdfMoments(cmd_args,collection="c0",kappa_models=("kappa","kappaBorn"),figname="pdfMoments",fontsize=22):

	#Moment labels
	moment_labels = (
		r"$\langle\kappa^2\rangle$",r"$\langle\vert\nabla\kappa\vert^2\rangle$",
		r"$\langle\kappa^3\rangle$",r"$\langle\kappa^2\nabla^2\kappa\rangle$",r"$\langle\vert\nabla\kappa\vert^2\nabla^2\kappa\rangle$",
		r"$\langle\kappa^4\rangle_c$",r"$\langle\kappa^3\nabla^2\kappa\rangle_c$",r"$\langle\kappa\vert\nabla\kappa\vert^2\nabla^2\kappa\rangle_c$",r"$\langle\vert\nabla\kappa\vert^4\rangle_c$"
		)
	
	#Model labels
	model_labels = {"kappa":"ray","kappaBorn":"born","kappaBornRT":"hybrid","kappaB+GP":"born+geo"}

	#Set up plot
	fig,axes = plt.subplots(3,3,figsize=(24,)*2)

	#Bootstrap mean to mimic LSST
	bootstrap_mean = lambda e: e.values.mean(0)

	#PDF for each kappa model
	for km in kappa_models:

		#Load samples
		fname = os.path.join(fiducial[collection].getMapSet(km).home,"convergence_moments_s50_nb9.npy")
		samples = Ensemble.read(fname).bootstrap(bootstrap_mean,bootstrap_size=1000,resample=1000)

		#Plot each bin
		for bn,ax in enumerate(axes.flatten()):
			ax.hist(samples[:,bn],bins=50,label=model_labels[km])

	#Axes labels
	for bn,ax in enumerate(axes.flatten()):
		ax.legend()
		ax.set_title(moment_labels[bn],fontsize=fontsize)

	#Scientific notation
	for ax in axes.flatten():
		ax.get_xaxis().get_major_formatter().set_powerlimits((0,0))
		plt.setp(ax.get_xticklabels(),rotation=30)

	#Save
	fig.tight_layout()
	fig.savefig(figname+".{0}".format(cmd_args.type))

###################################################################################################
###################################################################################################

def pbBias(cmd_args,feature_name="convergence_power_s0_nb100",title="Power spectrum",kappa_models=("Born",),callback=None,variation_idx=(0,),bootstrap_size=1000,resample=1000,return_results=False,fontsize=22):
	
	#Initialize plot
	fig,ax = plt.subplots(len(variation_idx),3,figsize=(24,8*len(variation_idx)))
	ax = np.atleast_2d(ax)

	##################
	#Load in the data#
	##################

	#Observation
	bootstrap_mean = lambda e: e.values.mean(0)
	feature_ray = Ensemble.read(os.path.join(fiducial["c0"].getMapSet("kappa").home,feature_name+".npy"),callback_loader=callback).bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample,seed=0)

	#Containers for cosmological model
	modelFeatures = dict()
	for mf in kappa_models:
		modelFeatures[mf] = dict()

	parameters = dict()

	for model in models:
		parameters[model.cosmo_id] = np.array([model.cosmology.Om0,model.cosmology.w0,model.cosmology.sigma8])
		for mf in kappa_models:

			try:
				modelFeatures[mf][model.cosmo_id] = Ensemble.read(os.path.join(model["c0"].getMapSet("kappa"+mf).home,feature_name+".npy"),callback_loader=callback)
			except IOError:
				pass

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
	
		feature_born = features[fiducial.cosmo_id].bootstrap(bootstrap_mean,bootstrap_size=bootstrap_size,resample=resample,seed=0)

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

			if return_results:
				assert len(kappa_models)==1
				assert len(variation_idx)==1

				return fitted_parameters_born,fitted_parameters_ray

			##########
			#Plotting#
			##########

			for n,p in enumerate(fisher.parameter_names):
				fitted_parameters_born[p].plot.hist(bins=50,ax=ax[nv,n],label=r"${\rm Control}$")
				fitted_parameters_ray[p].plot.hist(bins=50,ax=ax[nv,n],label=r"${\rm Observation}$")
				
				ax[nv,n].set_xlabel(plab[p],fontsize=fontsize)
				ax[nv,n].set_title(title)
				ax[nv,n].legend(mode="expand",ncol=2)

	#Labels
	for a in ax.flatten():
		plt.setp(a.get_xticklabels(),rotation=30)
	
	#Save
	fig.tight_layout()
	fig.savefig("bias_{0}.{1}".format(feature_name,cmd_args.type))

####################################################################################################################
####################################################################################################################

def pbBiasNgal(cmd_args,feature_names="convergence_powerSN{0}_s0_nb100",ngal=(10,15,20,30,40,50,60),kappa_model="Born",callback=None,variation_idx=0,bootstrap_size=1000,resample=1000,legend=True,title=r"$P^{\kappa\kappa}$",fontsize=22):
	
	#Set up plot
	fig,ax = plt.subplots()

	#Parameter placeholders
	lines = dict()

	#Cycle over ngal
	for ng in ngal:

		#Fit parameters with Born, ray
		pb,pr = pbBias(cmd_args,feature_name=feature_names.format(ng),
			kappa_models=(kappa_model,),callback=callback,variation_idx=(variation_idx,),bootstrap_size=bootstrap_size,
			resample=resample,return_results=True,fontsize=fontsize)

		#Add parameter
		for par in pb:
			if par not in lines:
				lines[par] = list()

		#Compute (pB-pR)/sigmaR
		for par in pb:
			bias = (pb[par] - pr[par]).mean()/pr[par].std()
			lines[par].append(bias)

	#Plot
	for par in lines:
		ax.plot(ngal,np.array(lines[par]),label=plab[par])

	#Legend
	ax.set_xlabel(r"$n_g({\rm arcmin}^{-2})$",fontsize=fontsize)
	ax.set_ylabel(r"$\langle p_{\rm born} - p_{\rm ray}\rangle/\sigma_{\rm ray}$")
	ax.set_title(title,fontsize=18)

	if legend:
		ax.legend(loc="upper right",prop={"size":17})

	#Save
	fig.tight_layout()
	fig.savefig("bias_ngal_{0}.{1}".format(feature_names.replace("{0}",""),cmd_args.type))

def pbBiasNgalPower(cmd_args):
	pbBiasNgal(cmd_args,feature_names="convergence_powerSN{0}_s0_nb100",title=r"$P^{\kappa\kappa}$")

def pbBiasNgalMoments(cmd_args):
	pbBiasNgal(cmd_args,feature_names="convergence_momentsSN{0}_s50_nb9",title=r"${\rm Moments}$",legend=False)

####################################################################################################################
####################################################################################################################
####################################################################################################################
####################################################################################################################

def pbBiasPowerSN15(cmd_args,feature_name="convergence_powerSN15_s0_nb100"):
	pbBias(cmd_args,feature_name=feature_name,title=r"$P^{\kappa\kappa}(n_g=15{\rm galaxies/arcmin}^2)$")

def pbBiasPowerSN30(cmd_args,feature_name="convergence_powerSN30_s0_nb100"):
	pbBias(cmd_args,feature_name=feature_name,title=r"$P^{\kappa\kappa}(n_g=30{\rm galaxies/arcmin}^2)$")

def pbBiasMomentsSN15(cmd_args,feature_name="convergence_momentsSN15_s50_nb9"):
	pbBias(cmd_args,feature_name=feature_name,kappa_models=("Born",),title=r"${\rm Moments}(n_g=15{\rm galaxies/arcmin}^2)$")

def pbBiasMomentsSN30(cmd_args,feature_name="convergence_momentsSN30_s50_nb9"):
	pbBias(cmd_args,feature_name=feature_name,title=r"${\rm Moments}(n_g=30{\rm galaxies/arcmin}^2)$")

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

def pbBiasPeaks(cmd_args,feature_name="convergence_peaks_s50_nb100"):
	callback = None
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

def pdfSkew(cmd_args):

	features = {
		r"$S_0({\rm noiseless})$" : (fiducial,"kappa","convergence_moments_s0_nb9",3),
		r"$S_0({\rm noiseless (Born)})$" : (fiducial,"kappaBorn","convergence_moments_s0_nb9",3),
		r"$S_0({\rm noiseless (RT)})$" : (fiducial,"kappaBornRT","convergence_moments_s0_nb9",3)
	}

	figname = "pdfSkew"

	pdfPlot(cmd_args,features=features,figname=figname)


####################################################################################################################

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()

method["1"] = convergenceVisualize
method["2"] = powerResiduals
method["3"] = plotSmoothSkew
method["3b"] = plotSmoothKurt
method["4"] = pdfMoments
method["5"] = pbBiasPowerSN30
method["5b"] = pbBiasMomentsSN15
method["5c"] = pbBiasMomentsSN30
method["6"] = pbBiasNgalPower
method["6b"] = pbBiasNgalMoments

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()