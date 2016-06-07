###########################################################################
################Cut interpolate Lens Planes between redshifts##############
###########################################################################

from __future__ import division

import sys,os
import json

from lenstools.simulations.logs import logdriver,logstderr,peakMemory,peakMemoryAll

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.pipeline.settings import PlaneSettings
from lenstools.simulations import PotentialPlane

from lenstools.utils.misc import ApproxDict


import numpy as np
import astropy.units as u

#Orchestra director of the execution
def PlanesTFRExecution():

	script_to_execute = planesTFR
	settings_handler = PlaneSettingsTFR
	kwargs = {}

	return script_to_execute,settings_handler,kwargs

#Bucketing function
def bucket(z,sorted_z):
	
	if z<sorted_z[0]:
		return (0,)

	if z>sorted_z[-1]:
		return (len(sorted_z)-1,)

	for n in range(len(sorted_z)-1):
		if (z>=sorted_z[n]) and z<sorted_z[n+1]:
			return (n,n+1)


#######################################################
################Main execution#########################
#######################################################

def planesTFR(pool,batch,settings,node_id):

	#Safety check
	assert pool is None,"planesTFR must run on one processon only!"
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
	
	#Original plane set and target plane set
	plane_set = realization.getPlaneSet(settings.directory_name)
	target_plane_set = realization.getPlaneSet(settings.target_plane_set)

	#Read in the info file, that contains the number of the snapshot, redshift and comoving distance
	info_filename = os.path.join(plane_set.storage,"info.txt")
	logdriver.info("Reading plane set {0} info file at {1}".format(plane_set.name,info_filename))
	
	with open(info_filename,"r") as fp:

		snapshot_number = list()
		distance = list()
		redshift = list()

		#Parse each line
		while True:
			
			#Split the line in snapshot,distance,redshift
			line = fp.readline().rstrip("\n").split(",")
			snapshot_number.append(int(line[0].split("=")[1]))
		
			d,unit = line[1].split("=")[1].split(" ")
			if unit=="Mpc/h":
				distance.append(float(d)*model.Mpc_over_h)
			else:
				distance.append(float(d)*getattr(u,"unit"))

			redshift.append(float(line[2].split("=")[1]))

		#Convert into arrays
		snapshot_number = np.array(snapshot_number)
		distance = u.quantity.Quantity(distance)
		redshift = np.array(redshift)

		#Sort
		distance = distance[np.argsort(redshift)]
		snapshot_number = snapshot_number[np.argsort(redshift)]
		redshift = np.sort(redshift)

	#Parse the cur2target mapping
	cur2target_filename = os.path.join(target_plane_set.home,settings.cur2target)
	logdriver.info("Reading redshift mapping from {0}".format(cur2target_filename))
	with open(cur2target_filename,"r") as fp:
		cur2target_parsed = json.load(fp)
		cur2target = ApproxDict((float(z),cur2target_parsed[z]) for z in cur2target)

	#Cycle over planes: for each plane, interpolate its values to the target redshift
	info_filename = os.path.join(target_plane_set.storage,"info.txt")
	logdriver.info("Writing target plane set {0} info file to {1}".format(plane_set.name,info_filename))
	with open(info_filename,"w") as fp:

		for n,nsnap in enumerate(snapshot_number):
			
			#Log
			target_redshift = cur2target[redshift[n]]
			logdriver.info("Redshift {0:.4f} will be mapped to redshift {1:.4f}".format(redshift[n],target_redshift))

			#Write corresponding line in the info file
			fp.write("s={0},d={1},z={2}\n".format(nsnap,distance[n],target_redshift))

			#Find the correct bucket
			z_indices = bucket(target_redshift,redshift)

			#Interpolate each cut point and normal
			for ncut,cut in settings.cut_points:
				for normal in settings.normals:

					target_plane_name = os.path.join(target_plane_set.storage,settings.name_format.format(nsnap,settings.kind,ncut,normal,settings.format))

					if len(z_indices)==2:

						#If inside the interpolation range, interpolate lensing density linearly
						source_plane_name1 = os.path.join(plane_set.storage,settings.name_format.format(snapshot_number[z_indices[0]],settings.kind,ncut,normal,settings.format))
						source_plane_name2 = os.path.join(plane_set.storage,settings.name_format.format(snapshot_number[z_indices[1]],settings.kind,ncut,normal,settings.format))
						
						logdriver.info("Reading plane at {0}".format(source_plane_name1))
						source_plane1 = PotentialPlane.load(source_plane_name1)
						logdriver.info("Reading plane at {0}".format(source_plane_name2))
						source_plane2 = PotentialPlane.load(source_plane_name2)

						#Interpolate to target redshift
						logdriver.info("Interpolating target redshift {0} between {1}-{2}".format(target_redshift,source_plane1.redshift,source_plane2.redshift))

						#Perform linear interpolation
						z1 = source_plane1.redshift
						z2 = source_plane2.redshift
						
						#Safety check
						assert z2>z1
						
						#Interpolate
						t = (target_redshift - z1) / (z2 - z1)
						source_plane1.data = (1-t)*source_plane1.data + t*source_plane2.data 

					
					elif len(z_indices)==1:

						#If out of the interpolation range, do nothing
						source_plane_name = os.path.join(plane_set.storage,settings.name_format.format(snapshot_number[z_indices[0]],settings.kind,ncut,normal,settings.format)) 
						logdriver.info("Target redshift {0} is out of interpolation range, plane is left unaltered")
						logdriver.info("Reading plane at {0}".format(source_plane_name))
						source_plane1 = PotentialPlane.load(source_plane_name)

					else:
						raise ValueError


					if settings.with_scale_factor:
						source_plane1.data *= (1.+target_redshift) / (1.+redshift[n])

					#Save to disk
					logdriver.info("Saving scaled plane to {0}".format(target_plane_name))
					source_plane1.redshift = target_redshift
					source_plane1.save(target_plane_name)


	#Done
	if pool is None or pool.is_master():
		logdriver.info("DONE!!")


#######################################################
################PlaneSettingsTFR#######################
#######################################################

class PlaneSettingsTFR(PlaneSettings):
	
	_section = "PlaneSettingsTFR"

	@classmethod
	def get(cls,options):

		#Section name
		section = cls._section
		
		#Parent constructor + additional attributes
		settings = super(PlaneSettingsTFR,cls).get(options)
		settings.parent_plane_set = options.get(section,"parent_plane_set")
		settings.cur2target = options.get(section,"cur2target")
		settings.with_scale_factor = options.getboolean(section,"with_scale_factor")

		#Return to user
		return settings

