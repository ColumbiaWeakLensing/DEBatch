#!/usr/bin/env python
from __future__ import division

import sys,os,glob
import logging
import json
import argparse

from lenstools.image.convergence import ConvergenceMap
from lenstools.image.noise import GaussianNoiseGenerator

from lenstools.statistics.ensemble import Ensemble
from lenstools.pipeline.simulation import SimulationBatch

import numpy as np
import astropy.units as u
from mpi4py import MPI

from emcee.utils import MPIPool

#############################################################################################
##############Measure the power spectrum#####################################################
#############################################################################################

def convergence_power(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"num_ell_nb{0}.npy".format(len(l_edges)-1)),conv.countModes(l_edges))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)

		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		l,Pl = conv.powerSpectrum(l_edges)
		return Pl

	except IOError:
		return None

##########################################################################################
##############Measure the cross power#####################################################
##########################################################################################

def cross_powerGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaGP",fnrep=("born_z","postBorn2-gp_z")):

	try:
		conv = ConvergenceMap.load(map_set.path(fname))
		conv2 = ConvergenceMap.load(map_set.path(fname).replace(map_set.name,cross).replace(*fnrep))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"num_ell_nb{0}.npy".format(len(l_edges)-1)),conv.countModes(l_edges))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)
			conv2 = conv2 + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)

		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")
			conv2 = conv2.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		l,Pl = conv.cross(conv2,l_edges=l_edges)
		return Pl

	except IOError:
		return None

def cross_powerLL(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaLL",fnrep=("born_z","postBorn2-ll_z")):
	return cross_powerGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=add_shape_noise,ngal=ngal,smoothing=smoothing,cross=cross,fnrep=fnrep)

##############################################################################
##############Peak counts#####################################################
##############################################################################

def convergence_peaks(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"th_peaks_nb{0}.npy".format(len(kappa_edges)-1)),0.5*(kappa_edges[1:]+kappa_edges[:-1]))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)

		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		k,peaks = conv.peakCount(kappa_edges)
		return peaks

	except IOError:
		return None


##########################################################################
##############Moments#####################################################
##########################################################################

def convergence_moments(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)
	
		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		#Subtract mean
		conv.data -= conv.data.mean()

		return conv.moments(connected=True)

	except IOError:
		return None

def cross_skewGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaGP",fnrep=("born_z","postBorn2-gp_z")):

	try:
		conv = ConvergenceMap.load(map_set.path(fname))
		conv2 = ConvergenceMap.load(map_set.path(fname).replace(map_set.name,cross).replace(*fnrep))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)
			conv2 = conv2 + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)

		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")
			conv2 = conv2.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		#Subtract mean
		conv.data -= conv.data.mean()
		conv2.data -= conv2.data.mean()

		#Measure cross skewness
		return np.array([(3*(conv.data**2)*(conv2.data)).mean()])

	except IOError:
		return None

def cross_skewLL(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaLL",fnrep=("born_z","postBorn2-ll_z")):
	return cross_skewGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=add_shape_noise,ngal=ngal,smoothing=smoothing,cross=cross,fnrep=fnrep)

def cross_kurtGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaGP",fnrep=("born_z","postBorn2-gp_z")):

	try:
		conv = ConvergenceMap.load(map_set.path(fname))
		conv2 = ConvergenceMap.load(map_set.path(fname).replace(map_set.name,cross).replace(*fnrep))
	
		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)
			conv2 = conv2 + gen.getShapeNoise(z=z,ngal=ngal*(u.arcmin**-2),seed=hash(os.path.basename(fname))%4294967295)

		if smoothing>0.:
			conv = conv.smooth(smoothing*u.arcmin,kind="gaussianFFT")
			conv2 = conv2.smooth(smoothing*u.arcmin,kind="gaussianFFT")

		#Subtract mean
		conv.data -= conv.data.mean()
		conv2.data -= conv2.data.mean()

		#Measure connected cross kurtosis
		kurt = 4*((conv.data**3)*conv2.data).mean() - 12*((conv.data**2).mean())*(conv.data*conv2.data).mean()
		return np.array([kurt])

	except IOError:
		return None

def cross_kurtLL(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,ngal=15,smoothing=0.0,cross="kappaLL",fnrep=("born_z","postBorn2-ll_z")):
	return cross_kurtGP(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=add_shape_noise,ngal=ngal,smoothing=smoothing,cross=cross,fnrep=fnrep)

#################################################################################
##############Main execution#####################################################
#################################################################################

if __name__=="__main__":

	#Command line argiments
	parser = argparse.ArgumentParser()
	parser.add_argument("-c","--config",dest="config",action="store",default=None,help="config file")
	parser.add_argument("-s","--smooth",dest="smooth",action="store",type=float,default=None,help="smoothing scale in arcmin")
	parser.add_argument("-N","--ngal",dest="ngal",action="store",type=int,default=None,help="number of galaxies per sq arcmin")
	parser.add_argument("-m","--maps",dest="maps",action="store",default=None,help="map sets file")
	parser.add_argument("-C","--collection",dest="collection",action="store",default="c0",help="model collection")
	cmd_args = parser.parse_args()

	#Make sure config is provided
	if (cmd_args.config is None) or (cmd_args.maps is None):
		parser.print_help()
		sys.exit(1)

	#Globals
	logging.basicConfig(level=logging.INFO)
	measurers = globals()

	#Read options from json config file
	with open(cmd_args.config,"r") as fp:
		options = json.load(fp)

	#Read the json map sets file
	with open(cmd_args.maps,"r") as fp:
		maps = json.load(fp)

	#Initialize MPIPool
	try:
		pool = MPIPool()
	except:
		pool = None

	if (pool is not None) and not(pool.is_master()):
		
		pool.wait()
		pool.comm.Barrier()
		MPI.Finalize()
		sys.exit(0)

	#Redshift
	redshift = options["redshift"]
	add_shape_noise = options["add_shape_noise"]

	#Smoothing
	smoothing = options["smoothing"]
	if cmd_args.smooth is not None:
		smoothing = cmd_args.smooth

	#Ngal per arcmin
	ngal = options["ngal"]
	if cmd_args.ngal is not None:
		ngal = cmd_args.ngal

	#Savename
	savename = options["method"]
	if add_shape_noise:
		savename += "SN{0}".format(ngal)

	#What to measure
	try:
		l_edges = np.linspace(*options["multipoles"])
	except TypeError:
		l_edges = None

	try:
		kappa_edges = np.linspace(*options["kappa_thresholds"])
	except TypeError:
		kappa_edges = None

	#How many chunks
	chunks = options["chunks"]

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring {0} for simulation batch at {1}".format(options["method"],batch.environment.home))

	#Save for reference
	if l_edges is not None:
		np.save(os.path.join(batch.home,"ell_nb{0}.npy".format(len(l_edges)-1)),0.5*(l_edges[1:]+l_edges[:-1]))
	
	if kappa_edges is not None:
		np.save(os.path.join(batch.home,"kappa_nb{0}.npy".format(len(kappa_edges)-1)),0.5*(kappa_edges[1:]+kappa_edges[:-1]))

	
	#####################################################################################################################

	for cosmo_id in maps:

		model = batch.getModel(cosmo_id)

		#Perform the measurements for all the map sets
		for ms in maps[cosmo_id]:

			map_set = model[cmd_args.collection].getMapSet(ms)

			#Log to user
			logging.info("Processing model {0}, map set {1}".format(map_set.cosmo_id,map_set.settings.directory_name))

			#Construct an ensemble for each map set
			ensemble_all = list()

			#Measure the descriptors spreading calculations on a MPIPool
			map_files = glob.glob(os.path.join(map_set.storage,"*.fits"))
			num_realizations = len(map_files)
			realizations_per_chunk = num_realizations // chunks

			for c in range(chunks):
				ensemble_all.append(Ensemble.compute(map_files[realizations_per_chunk*c:realizations_per_chunk*(c+1)],callback_loader=measurers[options["method"]],pool=pool,map_set=map_set,l_edges=l_edges,kappa_edges=kappa_edges,z=redshift,smoothing=smoothing,add_shape_noise=add_shape_noise,ngal=ngal))

			#Merge all the chunks
			ensemble_all = Ensemble.concat(ensemble_all,axis=0,ignore_index=True)

			#Save to disk
			ensemble_filename = os.path.join(map_set.home_subdir,savename+"_s{0}_nb{1}.npy".format(int(smoothing*100),ensemble_all.shape[1]))
			logging.info("Writing {0}".format(ensemble_filename))
			np.save(ensemble_filename,ensemble_all.values)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





