#######################################################################
################Cut lens planes out of a Gadget2 snapshot##############
#######################################################################

from __future__ import division

import sys,os
import json

from lenstools.simulations.logs import logdriver,logstderr,peakMemory,peakMemoryAll

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.pipeline.settings import PlaneSettings
from lenstools.simulations import PotentialPlane

from lenstools.utils import MPIWhirlPool

import numpy as np

#Orchestra director of the execution
def PlanesTFRExecution():

	script_to_execute = planesTFR
	settings_handler = PlaneSettingsTFR
	kwargs = {}

	return script_to_execute,settings_handler,kwargs

#Bucketing function
def bucket(z,sorted_z):
	pass

#######################################################
################Main execution#########################
#######################################################

def planesTFR(pool,batch,settings,node_id):

	#Safety check
	assert isinstance(pool,MPIWhirlPool) or (pool is None)
	assert isinstance(batch,SimulationBatch)
	assert isinstance(settings,PlaneSettingsTFR)

	#Split the id into the model,collection and realization parts
	cosmo_id,geometry_id,realization_id = node_id.split("|")

	#Get a handle on the simulation model
	model = batch.getModel(cosmo_id)

	#Get the realization number
	ic = int(realization_id.strip("ic"))

	#Instantiate the appropriate SimulationIC,SimulationPlanes objects 
	collection = model.getCollection(geometry_id)
	realization = collection.getRealization(ic)
	plane_set = realization.getPlaneSet(settings.directory_name)

	#Done
	if pool is None or pool.is_master():
		logdriver.info("DONE!!")


#######################################################
################PlaneSettingsTFR#######################
#######################################################

class PlaneSettingsTFR(PlaneSettings):
	pass
