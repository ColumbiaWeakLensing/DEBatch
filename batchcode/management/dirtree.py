#!/usr/bin/env python

import sys,os
import json
import itertools

import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value

sys.modules["mpi4py"] = None

import lenstools

from lenstools.pipeline.simulation import LensToolsCosmology
from lenstools.pipeline.settings import *
from lenstools.simulations.camb import CAMBSettings,parseLog
from lenstools.simulations.gadget2 import Gadget2Settings
from lenstools.simulations.design import Design

from batchcode.library.featureDB import DESimulationBatch
from batchcode.library.cutplanesTFR import PlaneSettingsTFR

#####################################################################

class DirTree(object):

	@property
	def order(self):
		return ["init","camb_linear","pfiles","camb_nonlinear","tfr"]


	def __init__(self):

		#Fixed options
		self.zmax = 3.1
		self.box_size_Mpc_over_h = 260.0
		self.nside = 512
		self.lens_thickness_Mpc = 120.0

		###############
		###Settings####
		###############

		#CAMB
		self.camb_lin = CAMBSettings()
		self.As_lin = self.camb_lin.scalar_amplitude
		self.camb_nl = CAMBSettings()
		self.camb_nl.do_nonlinear = 1

		#NGenIC
		self.ngenic = NGenICSettings()
		self.ngenic.GlassFile = lenstools.data("dummy_glass_little_endian.dat")
		self.seed = 5616559

		#Gadget2
		self.gadget2 = Gadget2Settings()
		self.gadget2.NumFilesPerSnapshot = 16

		#Planes
		self.planes = PlaneSettings.read("planes.ini")
		self.planesGeometry = PlaneSettingsTFR.read("planes_geometry_only.ini")
		self.planesGrowth = PlaneSettingsTFR.read("planes_growth_only.ini")

		#Maps/catalogs
		self.shear = CatalogSettings.read("catalog.ini")
		self.shearGeometry = CatalogSettings.read("catalog_geometry_only.ini")
		self.shearGrowth = CatalogSettings.read("catalog_growth_only.ini")

		#Init batch
		self.batch = DESimulationBatch.current()

		#################
		###File names####
		#################

		self.design_filename = "par_test.pkl"
		self.camb_lin_fileroot = "camb/camb_lin"
		self.camb_nl_fileroot = "camb/camb_nl"
		self.transfer_filename = "transfer_nl.pkl"
		self.redshift_mapping = "cur2target.json"



	###############################
	#Initialize the directory tree#
	###############################

	def init(self):

		##############################
		#First the fiducial cosmology#
		##############################

		model = self.batch.newModel(self.batch.fiducial_cosmology,self.batch._parameters)
		collection = model.newCollection(box_size=self.box_size_Mpc_over_h*model.Mpc_over_h,nside=self.nside)
		r = collection.newRealization(self.seed)
		pln_fiducial = r.newPlaneSet(self.planes)
		shearprod = collection.newCatalog(self.shear)

		#Directory dedicated to CAMB products
		collection.mkdir("camb")

		##############################
		#Next the non_fiducial models#
		##############################

		#Fisher variation models#
		for cosmo in self.batch.fisher_variation_cosmology:

			model = self.batch.newModel(cosmo,parameters=self.batch._parameters)
			collection = model.newCollection(box_size=self.box_size_Mpc_over_h*model.Mpc_over_h,nside=self.nside)
			r = collection.newRealization(self.seed)

			#Planes (native and from fiducial cosmology)
			pln = r.newPlaneSet(self.planes)
			r.linkPlaneSet(pln_fiducial,"Planes_fiducial")

			#Planes (geometry only and growth only)
			for settings in (self.planesGeometry,self.planesGrowth):
				pln = r.newPlaneSet(settings)

			#Catalogs
			for settings in (self.shear,self.shearGeometry,self.shearGrowth):  
				shearprod = collection.newCatalog(settings)

			#Directory dedicated to CAMB products
			collection.mkdir("camb")


	########################
	####CAMB linear mode####
	########################

	def camb_linear(self):

		#CAMB settings (linear run)
		for model in self.batch.models:
			model["c0"].writeCAMB(z=0.,settings=self.camb_lin,fname=self.camb_lin_fileroot+".param",output_root=self.camb_lin_fileroot)

	##############################################################
	##Comoving distances of the lenses and Gadget snapshot times##
	##############################################################

	def pfiles(self):

		#Compute comoving distance to maximum redshift for each model
		d = list()
		for model in self.batch.models:
			d.append(model.cosmology.comoving_distance(self.zmax))

		#Compute lens spacings
		d = np.array([dv.value for dv in d]) * d[0].unit

		#We want to make sure there are lenses up to the maximum of these distances
		lens_distances = np.arange(self.lens_thickness_Mpc,d.max().to(u.Mpc).value + self.lens_thickness_Mpc,self.lens_thickness_Mpc) * u.Mpc

		for model in self.batch.models:

			#Compute the redshifts of the Gadget snapshots
			z = np.zeros_like(lens_distances.value)
			for n,dlens in enumerate(lens_distances):
				z[n] = z_at_value(model.cosmology.comoving_distance,dlens)

			#Assgn values to gadget settings
			self.gadget2.OutputScaleFactor = np.sort(1/(1+z))

			#Write parameter files
			collection = model["c0"]

			#Convert camb power spectra into ngenic ones
			collection.camb2ngenic(z=0.0,input_root=self.camb_lin_fileroot)

			#NGenIC and Gadget2 parameter files
			r = collection["r0"]
			r.writeNGenIC(self.ngenic)
			r.writeGadget2(self.gadget2)

	#############################################################################################
	#####CAMB nonlinear mode: redshift scalings for geometry only and growth only cases##########
	#############################################################################################

	def camb_nonlinear(self):
		
		#Fiducial redshifts
		fiducial_model = self.batch.fiducial_model
		fiducial_a = fiducial_model["c0r0"].gadget_settings.OutputScaleFactor
		fiducial_z = 1/fiducial_a - 1

		#All the redshifts we need to compute the nonlinear transfer function at 
		all_fiducial_z = [fiducial_z]

		#All other model redshifts
		for model in self.batch.fisher_variation_models:
			
			collection = model["c0"]
			model_a = collection["r0"].gadget_settings.OutputScaleFactor
			model_z = 1/model_a - 1

			#Append to all fiducial z
			all_fiducial_z.append(model_z)

			#Parse the CAMB linear log to scale As to the appropriate value for the right sigma8
			camb_lin_filename = os.path.join(collection.home,self.camb_lin_fileroot+".out")
			camb_lin_log = parseLog(camb_lin_filename)
			sigma8 = camb_lin_log["sigma8"][0.0]
			new_As = self.As_lin*np.sqrt(collection.cosmology.sigma8/sigma8)
			print("[+] Parsed camb log at {0}: scale As to {1:3e} for correct sigma8(z=0)={2:.3f}".format(camb_lin_filename,new_As,collection.cosmology.sigma8))
			self.camb_nl.scalar_amplitude = new_As

			#Print the CAMB parameter file for the non linear calculations of the transfer function
			collection.writeCAMB(z=np.concatenate((fiducial_z,model_z)),settings=self.camb_nl,fname=self.camb_nl_fileroot+".param",output_root=self.camb_nl_fileroot)

			#First build dictionary for the growth only: each redshift needs to be associated to the one in the fiducial cosmology at the same comoving distance
			cur2target = dict()
			for n in range(len(model_z)):
				cur2target[model_z[n]] = fiducial_z[n]

			#Save the dictionary mapping
			dump_filename = os.path.join(collection["r0"].getPlaneSet("Planes_growth").home,self.redshift_mapping) 
			with open(dump_filename,"w") as fp:
				json.dump(cur2target,fp)
			print("[+] Dumped redshift mapping (Growth only) to {0}".format(dump_filename))

			#Next build dictionary for the geometry only: we are using the planes in the fiducial cosmology, but scaled to the corresponding redshift in the current cosmology
			cur2target = dict()
			for n in range(len(model_z)):
				cur2target[fiducial_z[n]] = model_z[n]

			#Save the dictionary mapping
			dump_filename = os.path.join(collection["r0"].getPlaneSet("Planes_geometry").home,self.redshift_mapping) 
			with open(dump_filename,"w") as fp:
				json.dump(cur2target,fp)
			print("[+] Dumped redshift mapping (Geometry only) to {0}".format(dump_filename))

		
		#Parse the CAMB linear log to scale As to the appropriate value for the right sigma8
		collection = fiducial_model["c0"]
		camb_lin_filename = os.path.join(collection.home,self.camb_lin_fileroot+".out")
		camb_lin_log = parseLog(camb_lin_filename)
		sigma8 = camb_lin_log["sigma8"][0.0]
		new_As = self.As_lin*np.sqrt(collection.cosmology.sigma8/sigma8)
		print("[+] Parsed camb log at {0}: scale As to {1:3e} for correct sigma8(z=0)={2:.3f}".format(camb_lin_filename,new_As,collection.cosmology.sigma8))
		self.camb_nl.scalar_amplitude = new_As

		#Print the CAMB parameter file for the non linear calculations of the transfer function
		collection.writeCAMB(z=np.concatenate(all_fiducial_z),settings=self.camb_nl,fname=self.camb_nl_fileroot+".param",output_root=self.camb_nl_fileroot)


	##############################################
	#####CAMB nonlinear transfer function#########
	##############################################

	def tfr(self):

		#######################################################################	
		#Load transfer function output from CAMB and compress it in a pkl file#
		#######################################################################

		for model in itertools.chain([self.batch.fiducial_model],self.batch.fisher_variation_models):
			
			collection = model["c0"] 
			tfr = collection.loadTransferFunction(self.camb_nl_fileroot+"_matterpower")
			transfer_savename = os.path.join(collection.home,self.transfer_filename) 
			tfr.save(transfer_savename)
			print("[+] Pickled non linear transfer function to {0}".format(transfer_savename))

		##################
		##Symbolic links##
		##################

		fiducial_model = self.batch.fiducial_model

		#Link the fiducial transfer function to ShearGeometry
		for model in self.batch.fisher_variation_models:
			
			collection = model["c0"]
			
			#Link the fiducial transfer function to ShearGeometry
			source = os.path.join(fiducial_model["c0"].home,self.transfer_filename)
			destination = os.path.join(collection.getCatalog("ShearGeometry").home,self.transfer_filename)
			os.symlink(source,destination)
			print("[+] Symlinked transfer function at {0} to {1}".format(source,destination))

			#Link the non fiducial transfer function to ShearGrowth
			source = os.path.join(collection.home,self.transfer_filename)
			destination = os.path.join(collection.getCatalog("ShearGrowth").home,self.transfer_filename)
			os.symlink(source,destination)
			print("[+] Symlinked transfer function at {0} to {1}".format(source,destination))


###########
#Execution#
###########

def main():
	tree = DirTree()
	for step in tree.order:
		if "--"+step in sys.argv:
			getattr(tree,step)()

if __name__=="__main__":
	main()







