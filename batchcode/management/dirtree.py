#!/usr/bin/env python

import sys,os
import json

import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value

sys.modules["mpi4py"] = None

import lenstools

from lenstools.pipeline.simulation import LensToolsCosmology
from lenstools.pipeline.settings import *
from lenstools.simulations.camb import CAMBSettings
from lenstools.simulations.gadget2 import Gadget2Settings
from lenstools.simulations.design import Design

from batchcode.lib.featureDB import DESimulationBatch
from batchcode.lib.rayGeomVGrowth import GVGMapSettings



#Fixed options
zmax = 3.1
box_size_Mpc_over_h = 260.0
nside = 512
lens_thickness_Mpc = 120.0

###############
###Settings####
###############

#CAMB
camb_lin = CAMBSettings()
camb_nl = CAMBSettings()
As_lin = camb_lin.scalar_amplitude

#NGenIC
ngenic = NGenICSettings()
ngenic.GlassFile = lenstools.data("dummy_glass_little_endian.dat")
seed = 5616559

#Gadget2
gadget2 = Gadget2Settings()
gadget2.NumFilesPerSnapshot = 24

#Planes
planes = PlaneSettings.read("planes.ini")

#Maps/catalogs
shear = MapSettings.read("maps.ini")
shearGeometry = GVGMapSettings.read("maps_geometry_only.ini")
shearGrowth = GVGMapSettings.read("maps_growth_only.ini")

#Init batch
batch = DESimulationBatch.current()

###############################
#Initialize the directory tree#
###############################

if "--init" in sys.argv:

	##############################
	#First the fiducial cosmology#
	##############################

	model = batch.newModel(batch.fiducial_cosmology,batch._parameters)
	collection = model.newCollection(box_size=box_size_Mpc_over_h*model.Mpc_over_h,nside=nside)
	r = collection.newRealization(seed)
	pln_fiducial = r.newPlaneSet(planes)
	shearprod = collection.newMapSet(shear)

	##############################
	#Next the non_fiducial models#
	##############################

	design = Design.read(os.path.join(batch.home,"data","par.pkl"))[["Om","w0","wa","sigma8"]]

	for Om,w0,wa,si8 in design.values:
	
		#Lay down directory tree
		cosmo = LensToolsCosmology(Om0=Om,Ode0=1-Om,w0=w0,wa=wa,sigma8=si8)
		model = batch.newModel(cosmo,parameters=["Om","Ode","w","wa","si"])
		collection = model.newCollection(box_size=box_size_Mpc_over_h*model.Mpc_over_h,nside=nside)
		r = collection.newRealization(seed)

		#Planes (native and from fiducial cosmology)
		pln = r.newPlaneSet(planes)
		r.linkPlaneSet(pln_fiducial,"Planes_fiducial")

		#Maps
		for settings in (shear,shearGeometry,shearGrowth):  
			shearprod = collection.newMapSet(settings)

############
####CAMB####
############

if "--camb_lin" in sys.argv:

	#CAMB settings (linear run)
	for model in batch.models:
		model["c0"].writeCAMB(z=0.,settings=camb_lin,fname="camb_lin.param",output_root="camb_lin")


if "--camb_nl" in sys.argv:
	pass


if "--tfr" in sys.argv:

	#######################################################################	
	#Load transfer function output from CAMB and compress it in a pkl file#
	#######################################################################

	for model in batch.models:
		collection = model["c0"] 
		tfr = collection.loadTransferFunction("camb_nl_transferfunc")
		tfr.save(os.path.join(collection.home,"transfer_nl.pkl"))

	##################
	##Symbolic links##
	##################


##############################################################
##Comoving distances of the lenses and Gadget snapshot times##
##############################################################

if ("--lenses" in sys.argv) or ("--pfiles" in sys.argv):

	#Compute comoving distance to maximum redshift for each model
	d = list()
	for model in batch.models:
		d.append(model.cosmology.comoving_distance(zmax))

	#Compute lens spacings
	d = np.array([dv.value for dv in d]) * d[0].unit

	#We want to make sure there are lenses up to the maximum of these distances
	lens_distances = np.arange(lens_thickness_Mpc,d.max().to(u.Mpc).value + lens_thickness_Mpc,lens_thickness_Mpc) * u.Mpc

	for model in batch.models:

		#Compute the redshifts of the Gadget snapshots
		z = np.zeros_like(lens_distances.value)
		for n,dlens in enumerate(lens_distances):
			z[n] = z_at_value(model.cosmology.comoving_distance,dlens)

		#Assgn values to gadget settings
		gadget2.OutputScaleFactor = np.sort(1/(1+z))

		if "--pfiles" in sys.argv:
		
			collection = model["c0"]

			#Convert camb power spectra into ngenic ones
			collection.camb2ngenic(z=0.0,input_root="camb_lin")

			#NGenIC and Gadget2 parameter files
			r = collection["r0"]
			r.writeNGenIC(ngenic)
			r.writeGadget2(gadget2)

		else:
			print(gadget2.OutputScaleFactor)

########################################################################
#####Redshift scalings for geometry only and growth only cases##########
########################################################################

if "--gvg" in sys.argv:
	
	#Fiducial redshifts
	fiducial_model = batch.fiducial_model
	fiducial_a = fiducial_model["c0r0"].gadget_settings.OutputScaleFactor
	fiducial_z = 1/fiducial_a - 1

	#All other model redshifts
	for model in batch.non_fiducial_models:
		
		collection = model["c0"]
		model_a = collection["r0"].gadget_settings.OutputScaleFactor
		model_z = 1/model_a - 1

		#First build dictionary for the growth only: each redshift needs to be associated to the one in the fiducial cosmology at the same comoving distance
		cur2target = dict()
		for n in range(len(model_z)):
			cur2target[model_z[n]] = fiducial_z[n]

		#Save the dictionary mapping
		with open(os.path.join(collection.getMapSet("MapsGrowth").home,"cur2target.json")) as fp:
			json.dump(cur2target,fp)

		#Next build dictionary for the geometry only: we are using the planes in the fiducial cosmology, but scaled to the corresponding redshift in the current cosmology
		cur2target = dict()
		for n in range(len(model_z)):
			cur2target[fiducial_z[n]] = model_z[n]

		#Save the dictionary mapping
		with open(os.path.join(collection.getMapSet("MapsGeometry").home,"cur2target.json")) as fp:
			json.dump(cur2target,fp)



