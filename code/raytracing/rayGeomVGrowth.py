########################################################
############Ray Tracing scripts#########################
########################################################
from __future__ import division,with_statement

import sys,os
import time
import gc

from operator import add
from functools import reduce

from lenstools.simulations.logs import logdriver,logstderr,peakMemory,peakMemoryAll

from lenstools.utils.mpi import MPIWhirlPool

from lenstools.image.convergence import Spin0
from lenstools import ConvergenceMap,ShearMap
from lenstools.catalog import Catalog,ShearCatalog

from lenstools.simulations.raytracing import RayTracer,TransferSpecs
from lenstools.simulations.camb import CAMBTransferFunction

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.pipeline.settings import MapSettings

import numpy as np
import astropy.units as u

#Orchestra director of the execution
def GVGExecution():

	script_to_execute = singleRedshift
	settings_handler = GVGMapSettings
	kwargs = {}

	return script_to_execute,settings_handler,kwargs

################################################
#######Single redshift ray tracing##############
################################################

def singleRedshift(pool,batch,settings,node_id):

	#Safety check
	assert isinstance(pool,MPIWhirlPool) or (pool is None)
	assert isinstance(batch,SimulationBatch)

	parts = node_id.split("|")

	if len(parts)==2:

		assert isinstance(settings,GVGMapSettings)
	
		#Separate the id into cosmo_id and geometry_id
		cosmo_id,geometry_id = parts

		#Get a handle on the model
		model = batch.getModel(cosmo_id)

		#Get the corresponding simulation collection and map batch handlers
		collection = [model.getCollection(geometry_id)]
		map_batch = collection[0].getMapSet(settings.directory_name)
		cut_redshifts = np.array([0.0])

	else:
		raise NotImplementedError

	#Override the settings with the previously pickled ones, if prompted by user
	if settings.override_with_local:

		local_settings_file = os.path.join(map_batch.home_subdir,"settings.p")
		settings = MapSettings.read(local_settings_file)
		assert isinstance(settings,MapSettings)

		if (pool is None) or (pool.is_master()):
			logdriver.warning("Overriding settings with the previously pickled ones at {0}".format(local_settings_file))

	##################################################################
	##################Settings read###################################
	##################################################################

	#Set random seed to generate the realizations
	if pool is not None:
		np.random.seed(settings.seed + pool.rank)
	else:
		np.random.seed(settings.seed)

	#Read map angle,redshift and resolution from the settings
	map_angle = settings.map_angle
	source_redshift = settings.source_redshift
	resolution = settings.map_resolution

	if len(parts)==2:

		#########################
		#Use a single collection#
		#########################

		#Read the plane set we should use
		plane_set = (settings.plane_set,)

		#Randomization
		nbody_realizations = (settings.mix_nbody_realizations,)
		cut_points = (settings.mix_cut_points,)
		normals = (settings.mix_normals,)
		map_realizations = settings.lens_map_realizations

	else:
		raise NotImplementedError

	####################################################################################################

	#TODO: read in the transfer function information
	transfer = None

	####################################################################################################

	#Decide which map realizations this MPI task will take care of (if pool is None, all of them)
	try:
		realization_offset = settings.first_realization - 1
	except AttributeError:
		realization_offset = 0

	if pool is None:
		first_map_realization = 0 + realization_offset
		last_map_realization = map_realizations + realization_offset
		realizations_per_task = map_realizations
		logdriver.debug("Generating lensing map realizations from {0} to {1}".format(first_map_realization+1,last_map_realization))
	else:
		assert map_realizations%(pool.size+1)==0,"Perfect load-balancing enforced, map_realizations must be a multiple of the number of MPI tasks!"
		realizations_per_task = map_realizations//(pool.size+1)
		first_map_realization = realizations_per_task*pool.rank + realization_offset
		last_map_realization = realizations_per_task*(pool.rank+1) + realization_offset
		logdriver.debug("Task {0} will generate lensing map realizations from {1} to {2}".format(pool.rank,first_map_realization+1,last_map_realization))

	#Planes will be read from this path
	plane_path = os.path.join("{0}","ic{1}","{2}")

	if (pool is None) or (pool.is_master()):
		for c,coll in enumerate(collection):
			logdriver.info("Reading planes from {0}".format(plane_path.format(coll.storage_subdir,"-".join([str(n) for n in nbody_realizations[c]]),plane_set[c])))

	#Plane info file is the same for all collections
	if (not hasattr(settings,"plane_info_file")) or (settings.plane_info_file is None):
		info_filename = batch.syshandler.map(os.path.join(plane_path.format(collection[0].storage_subdir,nbody_realizations[0][0],plane_set[0]),"info.txt"))
	else:
		info_filename = settings.plane_info_file

	if (pool is None) or (pool.is_master()):
		logdriver.info("Reading lens plane summary information from {0}".format(info_filename))

	#Read how many snapshots are available
	with open(info_filename,"r") as infofile:
		num_snapshots = len(infofile.readlines())

	#Save path for the maps
	save_path = map_batch.storage_subdir

	if (pool is None) or (pool.is_master()):
		logdriver.info("Lensing maps will be saved to {0}".format(save_path))

	begin = time.time()

	#Log initial memory load
	peak_memory_task,peak_memory_all = peakMemory(),peakMemoryAll(pool)
	if (pool is None) or (pool.is_master()):
		logstderr.info("Initial memory usage: {0:.3f} (task), {1[0]:.3f} (all {1[1]} tasks)".format(peak_memory_task,peak_memory_all))

	#We need one of these for cycles for each map random realization
	for rloc,r in enumerate(range(first_map_realization,last_map_realization)):

		#Instantiate the RayTracer
		tracer = RayTracer()

		#Force garbage collection
		gc.collect()

		#Start timestep
		start = time.time()
		last_timestamp = start

		#############################################################
		###############Add the lenses to the system##################
		#############################################################

		#Open the info file to read the lens specifications (assume the info file is the same for all nbody realizations)
		infofile = open(info_filename,"r")

		#Read the info file line by line, and decide if we should add the particular lens corresponding to that line or not
		for s in range(num_snapshots):

			#Read the line
			line = infofile.readline().strip("\n")

			#Stop if there is nothing more to read
			if line=="":
				break

			#Split the line in snapshot,distance,redshift
			line = line.split(",")

			snapshot_number = int(line[0].split("=")[1])
		
			distance,unit = line[1].split("=")[1].split(" ")
			if unit=="Mpc/h":
				distance = float(distance)*model.Mpc_over_h
			else:
				distance = float(distance)*getattr(u,"unit")

			lens_redshift = float(line[2].split("=")[1])

			#Select the right collection
			for n,z in enumerate(cut_redshifts):
				if lens_redshift>=z:
					c = n

			#Randomization of planes
			nbody = np.random.randint(low=0,high=len(nbody_realizations[c]))
			cut = np.random.randint(low=0,high=len(cut_points[c]))
			normal = np.random.randint(low=0,high=len(normals[c]))

			#Log to user
			logdriver.debug("Realization,snapshot=({0},{1}) --> NbodyIC,cut_point,normal=({2},{3},{4})".format(r,s,nbody_realizations[c][nbody],cut_points[c][cut],normals[c][normal]))

			#Add the lens to the system
			logdriver.info("Adding lens at redshift {0}".format(lens_redshift))
			plane_name = batch.syshandler.map(os.path.join(plane_path.format(collection[c].storage_subdir,nbody_realizations[c][nbody],plane_set[c]),settings.plane_name_format.format(snapshot_number,cut_points[c][cut],normals[c][normal],settings.plane_format)))
			tracer.addLens((plane_name,distance,lens_redshift))

		#Close the infofile
		infofile.close()

		now = time.time()
		logdriver.info("Plane specification reading completed in {0:.3f}s".format(now-start))
		last_timestamp = now

		#Rearrange the lenses according to redshift and roll them randomly along the axes
		tracer.reorderLenses()

		now = time.time()
		logdriver.info("Reordering completed in {0:.3f}s".format(now-last_timestamp))
		last_timestamp = now

		#Start a bucket of light rays from a regular grid of initial positions
		b = np.linspace(0.0,map_angle.value,resolution)
		xx,yy = np.meshgrid(b,b)
		pos = np.array([xx,yy]) * map_angle.unit

		if settings.tomographic_convergence:
			raise NotImplementedError

		else:

			#Trace the ray deflections
			jacobian = tracer.shoot(pos,z=source_redshift,kind="jacobians",transfer=transfer)

			now = time.time()
			logdriver.info("Jacobian ray tracing for realization {0} completed in {1:.3f}s".format(r+1,now-last_timestamp))
			last_timestamp = now

			#Compute shear,convergence and omega from the jacobians
			if settings.convergence:
		
				convMap = ConvergenceMap(data=1.0-0.5*(jacobian[0]+jacobian[3]),angle=map_angle)
				savename = batch.syshandler.map(os.path.join(save_path,"WLconv_z{0:.2f}_{1:04d}r.{2}".format(source_redshift,r+1,settings.format)))
				logdriver.info("Saving convergence map to {0}".format(savename)) 
				convMap.save(savename)
				logdriver.debug("Saved convergence map to {0}".format(savename)) 

			##############################################################################################################################
	
			if settings.shear:
		
				shearMap = ShearMap(data=np.array([0.5*(jacobian[3]-jacobian[0]),-0.5*(jacobian[1]+jacobian[2])]),angle=map_angle)
				savename = batch.syshandler.map(os.path.join(save_path,"WLshear_z{0:.2f}_{1:04d}r.{2}".format(source_redshift,r+1,settings.format)))
				logdriver.info("Saving shear map to {0}".format(savename))
				shearMap.save(savename) 

			##############################################################################################################################
	
			if settings.omega:
		
				omegaMap = Spin0(data=-0.5*(jacobian[2]-jacobian[1]),angle=map_angle)
				savename = batch.syshandler.map(os.path.join(save_path,"WLomega_z{0:.2f}_{1:04d}r.{2}".format(source_redshift,r+1,settings.format)))
				logdriver.info("Saving omega map to {0}".format(savename))
				omegaMap.save(savename)

		now = time.time()
		
		#Log peak memory usage to stdout
		peak_memory_task,peak_memory_all = peakMemory(),peakMemoryAll(pool)
		logdriver.info("Weak lensing calculations for realization {0} completed in {1:.3f}s".format(r+1,now-last_timestamp))
		logdriver.info("Peak memory usage: {0:.3f} (task), {1[0]:.3f} (all {1[1]} tasks)".format(peak_memory_task,peak_memory_all))

		#Log progress and peak memory usage to stderr
		if (pool is None) or (pool.is_master()):
			logstderr.info("Progress: {0:.2f}%, peak memory usage: {1:.3f} (task), {2[0]:.3f} (all {2[1]} tasks)".format(100*(rloc+1.)/realizations_per_task,peak_memory_task,peak_memory_all))
	
	#Safety sync barrier
	if pool is not None:
		pool.comm.Barrier()

	if (pool is None) or (pool.is_master()):	
		now = time.time()
		logdriver.info("Total runtime {0:.3f}s".format(now-begin))


#################################################################################################################################

class GVGMapSettings(MapSettings):

	_section = "GVGMapSettings"

	def _read_transfer(self,options,section):

		self.tfr_filename = options.get(section,"tfr_filename")
		self.cur2target = options.get(section,"cur2target")
		self.with_scale_factor = options.getboolean(section,"with_scale_factor")
		self.scaling_method = options.get(section,"scaling_method")

	@classmethod
	def get(cls,options):
		settings = super(GVGMapSettings,cls).get(options)
		settings._read_transfer(options,cls._section)
		return settings





