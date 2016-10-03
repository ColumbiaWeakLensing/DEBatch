#!/usr/bin/env python-mpi
from __future__ import division

import sys,os
import logging

from lenstools.image.convergence import ConvergenceMap
from lenstools.statistics.ensemble import Ensemble
from lenstools.pipeline.simulation import SimulationBatch

import numpy as np
import astropy.units as u
from mpi4py import MPI

from emcee.utils import MPIPool

#############################################################################################
##############Measure the power spectrum#####################################################
#############################################################################################

def convergence_power(fname,map_set,l_edges,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"num_ell_nb{0}.npy".format(len(l_edges)-1)),conv.countModes(l_edges))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		l,Pl = conv.powerSpectrum(l_edges)
		return Pl

	except IOError:
		return None

##############################################################################
##############Peak counts#####################################################
##############################################################################

def convergence_peaks(fname,map_set,kappa_edges,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))

		if "0001r" in fname:
			np.save(os.path.join(map_set.home_subdir,"th_peaks_nb{0}.npy".format(len(kappa_edges)-1)),0.5*(kappa_edges[1:]+kappa_edges[:-1]))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		k,peaks = conv.peakCount(kappa_edges)
		return peaks

	except IOError:
		return None


##########################################################################
##############Moments#####################################################
##########################################################################

def convergence_moments(fname,map_set,smoothing_scale=0.0*u.arcmin):
	
	try:
		conv = ConvergenceMap.load(map_set.path(fname))
	
		if smoothing_scale>0:
			conv = conv.smooth(smoothing_scale,kind="gaussianFFT")

		return conv.moments(connected=True)

	except IOError:
		return None

#################################################################################
##############Main execution#####################################################
#################################################################################

if __name__=="__main__":

	logging.basicConfig(level=logging.INFO)

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
	redshift = 2.0

	#What to measure
	l_edges = np.arange(100,10000,100)
	kappa_edges = np.linspace(-.05,.5,101)

	#How many realizations
	num_realizations = 1024
	chunks = 16
	realizations_per_chunk = num_realizations // chunks

	#Get a handle on the simulation batch
	batch = SimulationBatch.current()
	logging.info("Measuring descriptors for simulation batch at {0}".format(batch.environment.home))

	#Save for reference
	np.save(os.path.join(collection.home_subdir,"ell_nb{0}.npy".format(len(l_edges)-1)),0.5*(l_edges[1:]+l_edges[:-1]))
	#np.save(os.path.join(collection.home_subdir,"kappa_nb{0}.npy".format(len(kappa_edges)-1)),0.5*(kappa_edges[1:]+kappa_edges[:-1]))

	for model in batch.models:

		#Perform the measurements for all the map sets
		for map_set in [model["c0"].mapsets]:

			#Log to user
			logging.info("Processing model {0}, map set {1}".format(map_set.cosmo_id,map_set.settings.directory_name))

			#Construct an ensemble for each map set
			ensemble_all = list()

			#Measure the descriptors spreading calculations on a MPIPool
			for c in range(chunks):
				ensemble_all.append(Ensemble.compute([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(realizations_per_chunk*c,realizations_per_chunk*(c+1)) ],callback_loader=convergence_power,pool=pool,map_set=map_set,l_edges=l_edges))
				#ensemble_all.append(Ensemble.compute([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(realizations_per_chunk*c,realizations_per_chunk*(c+1)) ],callback_loader=convergence_peaks,pool=pool,map_set=map_set,kappa_edges=kappa_edges))
				#ensemble_all.append(Ensemble.compute([ "WLconv_z{0:.2f}_{1:04d}r.fits".format(redshift,r+1) for r in range(realizations_per_chunk*c,realizations_per_chunk*(c+1)) ],callback_loader=convergence_moments,pool=pool,map_set=map_set))

			#Merge all the chunks
			ensemble_all = Ensemble.concat(ensemble_all,axis=0,ignore_index=True)

			#Save to disk
			savename = os.path.join(map_set.home_subdir,"power_s{0}_nb{1}.npy".format(0,ensemble_all.shape[1]))
			logging.info("Writing {0}".format(savename))
			np.save(savename,ensemble_all.values)

	#Close pool and quit
	if pool is not None:
		
		pool.close()
		pool.comm.Barrier()
		MPI.Finalize()
	
	sys.exit(0)





