########################################################
############Ray Tracing scripts#########################
########################################################
from __future__ import division,with_statement

import sys,os
import time
import gc
import json

from operator import add
from functools import reduce

from lenstools.simulations.logs import logdriver,logstderr,peakMemory,peakMemoryAll

from lenstools.utils.misc import ApproxDict
from lenstools.utils.mpi import MPIWhirlPool

from lenstools.image.convergence import Spin0
from lenstools import ConvergenceMap,ShearMap

from lenstools.simulations.raytracing import RayTracer,TransferSpecs
from lenstools.simulations.camb import CAMBTransferFunction

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.pipeline.settings import CatalogSettings

import numpy as np
import astropy.units as u

#Orchestra director of the execution
def GVGExecution():

	script_to_execute = simulatedCatalog
	settings_handler = GVGCatalogSettings
	kwargs = {}

	return script_to_execute,settings_handler,kwargs

################################################
#######Single redshift ray tracing##############
################################################

def simulatedCatalog(pool,batch,settings,node_id):

	#Safety check
	assert isinstance(pool,MPIWhirlPool) or (pool is None)
	assert isinstance(batch,SimulationBatch)
	assert isinstance(settings,GVGCatalogSettings)

	#Separate the id into cosmo_id and geometry_id
	cosmo_id,geometry_id = node_id.split("|")

	#Get a handle on the model
	model = batch.getModel(cosmo_id)

	#Get the corresponding simulation collection and catalog handler
	collection = model.getCollection(geometry_id)
	catalog = collection.getCatalog(settings.directory_name)

	#Override the settings with the previously pickled ones, if prompted by user
	if settings.override_with_local:

		local_settings_file = os.path.join(catalog.home_subdir,"settings.p")
		settings = CatalogSettings.read(local_settings_file)
		assert isinstance(settings,CatalogSettings)

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

	#Read the catalog save path from the settings
	catalog_save_path = catalog.storage_subdir
	if (pool is None) or (pool.is_master()):
		logdriver.info("Lensing catalogs will be saved to {0}".format(catalog_save_path))

	#Read the total number of galaxies to raytrace from the settings
	total_num_galaxies = settings.total_num_galaxies

	#Pre-allocate numpy arrays
	initial_positions = np.zeros((2,total_num_galaxies)) * settings.catalog_angle_unit
	galaxy_redshift = np.zeros(total_num_galaxies)

	#Keep track of the number of galaxies for each catalog
	galaxies_in_catalog = list()

	#Fill in initial positions and redshifts
	for galaxy_position_file in settings.input_files:

		try:
			galaxies_before = reduce(add,galaxies_in_catalog)
		except TypeError:
			galaxies_before = 0
	
		#Read the galaxy positions and redshifts from the position catalog
		if (pool is None) or (pool.is_master()):
			logdriver.info("Reading galaxy positions and redshifts from {0}".format(galaxy_position_file))
		
		position_catalog = Catalog.read(galaxy_position_file)

		if (pool is None) or (pool.is_master()):
			logdriver.info("Galaxy catalog {0} contains {1} galaxies".format(galaxy_position_file,len(position_catalog)))

		#This is just to avoid confusion
		assert position_catalog.meta["AUNIT"]==settings.catalog_angle_unit.to_string(),"Catalog angle units, {0}, do not match with the ones privided in the settings, {1}".format(position_catalog.meta["AUNIT"],settings.catalog_angle_unit.to_string())

		#Keep track of the number of galaxies in the catalog
		galaxies_in_catalog.append(len(position_catalog))

		if (pool is None) or (pool.is_master()):
			#Save a copy of the position catalog to the simulated catalogs directory
			position_catalog.write(os.path.join(catalog_save_path,os.path.basename(galaxy_position_file)),overwrite=True)

		#Fill in initial positions and redshifts
		initial_positions[0,galaxies_before:galaxies_before+len(position_catalog)] = position_catalog["x"] * getattr(u,position_catalog.meta["AUNIT"])
		initial_positions[1,galaxies_before:galaxies_before+len(position_catalog)] = position_catalog["y"] * getattr(u,position_catalog.meta["AUNIT"])
		galaxy_redshift[galaxies_before:galaxies_before+len(position_catalog)] = position_catalog["z"]

	#Make sure that the total number of galaxies matches, and units are correct
	assert reduce(add,galaxies_in_catalog)==total_num_galaxies,"The total number of galaxies in the catalogs, {0}, does not match the number provided in the settings, {1}".format(reduce(add,galaxies_in_catalog),total_num_galaxies)

	##########################################################################################################################################################
	####################################Initial positions and redshifts of galaxies loaded####################################################################
	##########################################################################################################################################################

	#Read the randomization information from the settings
	nbody_realizations = settings.mix_nbody_realizations
	cut_points = settings.mix_cut_points
	normals = settings.mix_normals
	catalog_realizations = settings.lens_catalog_realizations

	if hasattr(settings,"realizations_per_subdirectory"):
		realizations_in_subdir = settings.realizations_per_subdirectory
	else:
		realizations_in_subdir = catalog_realizations

	#Create subdirectories as necessary
	catalog_subdirectory = _subdirectories(catalog_realizations,realizations_in_subdir)
	if (pool is None) or (pool.is_master()):
		
		for d in catalog_subdirectory:
			
			dir_to_make = os.path.join(catalog_save_path,d)
			if not(os.path.exists(dir_to_make)):
				logdriver.info("Creating catalog subdirectory {0}".format(dir_to_make))
				os.mkdir(dir_to_make)

	#Safety barrier sync
	if pool is not None:
		pool.comm.Barrier() 

	####################################################################################################

	#Read in transfer function
	tfr_filename = os.path.join(catalog.home,settings.tfr_filename)
	z_mapping_filename = os.path.join(catalog.home,settings.cur2target)
	
	if pool is None or pool.is_master():
		logdriver.info("Read pickled CAMB transfer function from {0}".format(tfr_filename))
		logdriver.info("Read redshift mapping from {0}".format(z_mapping_filename)) 
	
	tfr = CAMBTransferFunction.read(tfr_filename)
	with open(z_mapping_filename,"r") as fp:
		mapping_json = json.load(fp)
		cur2target = ApproxDict((float(z),mapping_json[z]) for z in mapping_json)

	#If scaling is performed with FFTs, generate the k meshgrid
	if settings.scaling_method=="FFT":
		kx,ky = np.meshgrid(np.fft.rfftfreq(settings.fft_mesh_size),np.fft.fftfreq(settings.fft_mesh_size))
		kmesh = np.sqrt(kx**2+ky**2)*2.*np.pi / collection[0].box_size.to(u.Mpc)
	else:
		kmesh = None

	transfer = TransferSpecs(tfr,cur2target,settings.with_scale_factor,kmesh,settings.scaling_method)

	if pool is None or pool.is_master():
		logdriver.info("Density fluctuation scaling settings: with_scale_factor={0}, scaling_method={1}".format(transfer.with_scale_factor,transfer.scaling_method))

	####################################################################################################

	#Decide which map realizations this MPI task will take care of (if pool is None, all of them)
	try:
		realization_offset = settings.first_realization - 1
	except AttributeError:
		realization_offset = 0

	if pool is None:
		first_realization = 0 + realization_offset
		last_realization = catalog_realizations + realization_offset
		realizations_per_task = catalog_realizations
		logdriver.debug("Generating lensing catalog realizations from {0} to {1}".format(first_realization+1,last_realization))
	else:
		assert catalog_realizations%(pool.size+1)==0,"Perfect load-balancing enforced, catalog_realizations must be a multiple of the number of MPI tasks!"
		realizations_per_task = catalog_realizations//(pool.size+1)
		first_realization = realizations_per_task*pool.rank + realization_offset
		last_realization = realizations_per_task*(pool.rank+1) + realization_offset
		logdriver.debug("Task {0} will generate lensing catalog realizations from {1} to {2}".format(pool.rank,first_realization+1,last_realization))


	#Planes will be read from this path
	plane_path = os.path.join(collection.storage_subdir,"ic{0}",settings.plane_set)

	if (pool is None) or (pool.is_master()):
		logdriver.info("Reading planes from {0}".format(plane_path.format("-".join([str(n) for n in nbody_realizations]))))

	#Read how many snapshots are available
	with open(batch.syshandler.map(os.path.join(plane_path.format(nbody_realizations[0]),"info.txt")),"r") as infofile:
		num_snapshots = len(infofile.readlines())


	#Construct the randomization matrix that will differentiate between realizations; this needs to have shape map_realizations x num_snapshots x 3 (ic+cut_points+normals)
	randomizer = np.zeros((catalog_realizations,num_snapshots,3),dtype=np.int)
	randomizer[:,:,0] = np.random.randint(low=0,high=len(nbody_realizations),size=(catalog_realizations,num_snapshots))
	randomizer[:,:,1] = np.random.randint(low=0,high=len(cut_points),size=(catalog_realizations,num_snapshots))
	randomizer[:,:,2] = np.random.randint(low=0,high=len(normals),size=(catalog_realizations,num_snapshots))

	if (pool is None) or (pool.is_master()):
		logdriver.debug("Randomization matrix has shape {0}".format(randomizer.shape))


	begin = time.time()

	#Log initial memory load
	peak_memory_task,peak_memory_all = peakMemory(),peakMemoryAll(pool)
	if (pool is None) or (pool.is_master()):
		logstderr.info("Initial memory usage: {0:.3f} (task), {1[0]:.3f} (all {1[1]} tasks)".format(peak_memory_task,peak_memory_all))

	#We need one of these for cycles for each map random realization
	for rloc,r in enumerate(range(first_realization,last_realization)):

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
		infofile = open(os.path.join(plane_path.format(nbody_realizations[0]),"info.txt"),"r")

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

			#Add the lens to the system
			logdriver.info("Adding lens at redshift {0}".format(lens_redshift))
			plane_name = batch.syshandler.map(os.path.join(plane_path.format(nbody_realizations[randomizer[r,s,0]]),settings.plane_name_format.format(snapshot_number,cut_points[randomizer[r,s,1]],normals[randomizer[r,s,2]],settings.plane_format)))
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

		#Trace the ray deflections through the lenses
		jacobian = tracer.shoot(initial_positions,z=galaxy_redshift,kind="jacobians",transfer=transfer)

		now = time.time()
		logdriver.info("Jacobian ray tracing for realization {0} completed in {1:.3f}s".format(r+1,now-last_timestamp))
		last_timestamp = now

		#Build the shear catalog and save it to disk
		shear_catalog = ShearCatalog([0.5*(jacobian[3]-jacobian[0]),-0.5*(jacobian[1]+jacobian[2])],names=("shear1","shear2"))

		for n,galaxy_position_file in enumerate(settings.input_files):

			try:
				galaxies_before = reduce(add,galaxies_in_catalog[:n])
			except TypeError:
				galaxies_before = 0
		
			#Build savename
			if len(catalog_subdirectory):
				shear_catalog_savename = batch.syshandler.map(os.path.join(catalog_save_path,catalog_subdirectory[r//realizations_in_subdir],"WLshear_"+os.path.basename(galaxy_position_file.split(".")[0])+"_{0:04d}r.{1}".format(r+1,settings.format)))
			else:
				shear_catalog_savename = batch.syshandler.map(os.path.join(catalog_save_path,"WLshear_"+os.path.basename(galaxy_position_file.split(".")[0])+"_{0:04d}r.{1}".format(r+1,settings.format)))
			
			logdriver.info("Saving simulated shear catalog to {0}".format(shear_catalog_savename))
			shear_catalog[galaxies_before:galaxies_before+galaxies_in_catalog[n]].write(shear_catalog_savename,overwrite=True)

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

class GVGCatalogSettings(CatalogSettings):

	_section = "GVGCatalogSettings"

	@classmethod
	def get(cls,options):

		settings = super(GVGCatalogSettings,cls).get(options)
		settings.tfr_filename = options.get(cls._section,"tfr_filename")
		settings.cur2target = options.get(cls._section,"cur2target")
		settings.with_scale_factor = options.getboolean(cls._section,"with_scale_factor")
		settings.scaling_method = options.get(cls._section,"scaling_method")
		settings.fft_mesh_size = options.getint(cls._section,"fft_mesh_size")

		return settings





