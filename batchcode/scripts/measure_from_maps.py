#!/usr/bin/env python
from __future__ import division

import sys,os,glob
import logging
import json

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

def convergence_power(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"num_ell_nb{0}.npy".format(len(l_edges)-1)),conv.countModes(l_edges))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,seed=hash(os.path.basename(fname)))

		l,Pl = conv.powerSpectrum(l_edges)
		return Pl

	except IOError:
		return None

##############################################################################
##############Peak counts#####################################################
##############################################################################

def convergence_peaks(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"th_peaks_nb{0}.npy".format(len(kappa_edges)-1)),0.5*(kappa_edges[1:]+kappa_edges[:-1]))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,seed=hash(os.path.basename(fname)))

		k,peaks = conv.peakCount(kappa_edges)
		return peaks

	except IOError:
		return None


##########################################################################
##############Moments#####################################################
##########################################################################

def convergence_moments(fname,map_set,l_edges,kappa_edges,z,add_shape_noise=False,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		if add_shape_noise:
			gen = GaussianNoiseGenerator.forMap(conv)
			conv = conv + gen.getShapeNoise(z=z,seed=hash(os.path.basename(fname)))

		return conv.moments(connected=True)

	except IOError:
		return None

#################################################################################
##############Main execution#####################################################
#################################################################################

if __name__=="__main__":

	#Globals
	logging.basicConfig(level=logging.INFO)
	measurers = globals()

	#Read options from json config file
	with open(sys.argv[1],"r") as fp:
		options = json.load(fp)

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

	#Savename
	savename = options["method"]
	if add_shape_noise:
		savename += "SN"

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

	for model in batch.models:

		#Perform the measurements for all the map sets
		for map_set in model["c0"].mapsets:

			#Log to user
			logging.info("Processing model {0}, map set {1}".format(map_set.cosmo_id,map_set.settings.directory_name))

			#Construct an ensemble for each map set
			ensemble_all = list()

			#Measure the descriptors spreading calculations on a MPIPool
			map_files = glob.glob(os.path.join(map_set.storage,"*.fits"))
			num_realizations = len(map_files)
			realizations_per_chunk = num_realizations // chunks

			for c in range(chunks):
				ensemble_all.append(Ensemble.compute(map_files[realizations_per_chunk*c:realizations_per_chunk*(c+1)],callback_loader=measurers[options["method"]],pool=pool,map_set=map_set,l_edges=l_edges,kappa_edges=kappa_edges,z=redshift,add_shape_noise=add_shape_noise))

			#Merge all the chunks
			ensemble_all = Ensemble.concat(ensemble_all,axis=0,ignore_index=True)

			#Save to disk
			ensemble_filename = os.path.join(map_set.home_subdir,savename+"_s{0}_nb{1}.npy".format(0,ensemble_all.shape[1]))
			logging.info("Writing {0}".format(ensemble_filename))
			np.save(savename,ensemble_all.values)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





