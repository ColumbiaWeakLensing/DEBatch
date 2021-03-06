############################################
#Measurements of features on shear catalogs#
############################################

from __future__ import division,print_function,with_statement
import sys,os,re
from itertools import product

from lenstools.catalog import ShearCatalog
from lenstools.statistics.ensemble import Ensemble, SquareMatrix
from lenstools.statistics.database import Database
from lenstools.pipeline.simulation import SimulationBatch,LensToolsCosmology,string2cosmo

import numpy as np

try:
	import matplotlib.pyplot as plt
except:
	pass

import astropy.units as u
import astropy.table as tbl

real = re.compile(r'([0-9]+)r\.fits')

#############################################
#Measure features in a partcular realization#
#############################################

def process_realization(realization,db_type,map_specs,sub_catalog,measurer,**kwargs):

	"""
	:param realization: number of the realization to process
	:type realization: int.

	:param sub_catalog: sub catalog that the realization belongs to
	:type sub_catalog: SimulationSubCatalog

	:param measurer: gets called on the list of convergence maps in the particular realization, must return an Ensemble
	:type measurer: callable

	:param kwargs: passed to the measurer
	:type kwargs: dict.

	:returns: Ensemble

	"""

	#Construct file names of shear and position catalogs
	position_file_path = os.path.join(sub_catalog.environment.storage,sub_catalog.cosmo_id,sub_catalog.geometry_id,sub_catalog.name)
	position_files = [ os.path.join(position_file_path,"positions_bin{0}.fits".format(n)) for n in range(1,6) ]
	shear_files = [ os.path.join(sub_catalog.storage_subdir,"WLshear_positions_bin{0}_{1:04d}r.fits".format(n,realization)) for n in range(1,6) ]

	#Construct the kappa maps
	maps = db_type.make_maps(shear_files,position_files,**map_specs)

	#Measure the features
	ensemble_realization = measurer(maps,**kwargs)

	#Add the realization label
	ensemble_realization["realization"] = realization

	#Return 
	return ensemble_realization

#Aggregate results of computation
def _assemble(ens_list):
	return Ensemble.concat(ens_list,axis=0,ignore_index=True)

################################
#####FeatureDatabase class######
################################

class FeatureDatabase(Database):

	#Global options
	map_specs = {
	
	"npixel" : 512,
	"smooth" : 0.5*u.arcmin,
	"fov" : 3.5*u.deg,
	"zbins" : [(0.0052829915857917076, 0.46370802037163456), (0.46370802037163456, 0.68921284184762954),(0.68921284184762954, 0.93608623056054063),(0.93608623056054063, 1.2872107430836479),(1.2872107430836479, 2.9998163872653354)],
	"add_shape_noise" : False,
	"photoz_bias" : None,
	"photoz_sigma" : None,

	"to_split" : ["Om","w","wa"],
	"psplit" : "geometry"
	
	}


	def __init__(self,name,**kwargs):
		super(FeatureDatabase,self).__init__(name)
		for key in kwargs.keys():
			self.map_specs[key] = kwargs[key]

	###########################################################################################
	#Process all the realizations in a particular sub-catalog; add the results to the database#
	###########################################################################################

	def add_features(self,table_name,sub_catalog,measurer,extra_columns=None,pool=None,**kwargs):

		"""

		:param table_name: name of the SQL table to add the feature to
		:type table_name: str.

		:param sub_catalog: sub catalog that the realization belongs to
		:type sub_catalog: SimulationSubCatalog

		:param measurer: gets called on the list of convergence maps in the particular realization, must return an Ensemble
		:type measurer: callable

		:param extra_columns: dictionary whose keys are the names of the extra columns to insert in the database, and whose values are the column values for the sub_catalog
		:type extra_columns: dict.

		:param kwargs: passed to the measurer
		:type kwargs: dict.

		:param pool: MPIPool to spread the sub_catalog computations onto
		:type pool: MPIPool

		"""

		#First and last realization to process in the sub_catalog
		first_realization = sub_catalog.first_realization
		last_realization = sub_catalog.last_realization

		#Compute Ensemble of realizations
		ensemble_sub_catalog = Ensemble.compute(range(first_realization,last_realization+1),callback_loader=process_realization,assemble=_assemble,pool=pool,map_specs=self.map_specs,db_type=self.__class__,sub_catalog=sub_catalog,measurer=measurer,**kwargs)


		########################################################################################
		#Add the cosmological parameters as additional columns, but separate geometry VS growth#
		########################################################################################

		to_split = self.map_specs["to_split"] 
		if  to_split is None:
			
			#Nothing surprising if there is no parameter splitting
			for p in sub_catalog.environment.name2attr:
				pattr = sub_catalog.environment.name2attr[p]
				ensemble_sub_catalog[pattr]  = getattr(sub_catalog.cosmology,pattr)

		else:

			#Perform the parameter splitting
			for p in sub_catalog.environment.name2attr:
				pattr = sub_catalog.environment.name2attr[p]

				if p in self.map_specs["to_split"]:

					for split_type in ["geometry","growth"]:
						 ensemble_sub_catalog[pattr+"_"+split_type] = DESimulationBatch._fiducial_params[p]

					ensemble_sub_catalog[pattr+"_"+self.map_specs["psplit"]] = getattr(sub_catalog.cosmology,pattr)

				else:
					ensemble_sub_catalog[pattr]  = getattr(sub_catalog.cosmology,pattr)

		################################################################################################

		#Add the extra columns
		if extra_columns is not None:
			for key in extra_columns.keys():
				
				if key in ensemble_sub_catalog.columns:
					raise ValueError("Column {0} already present in database!".format(key))
				
				ensemble_sub_catalog[key] = extra_columns[key]

		#Insert into the database
		self.insert(ensemble_sub_catalog,table_name)


	###############################################
	#Create convergence maps out of shear catalogs#
	###############################################
	
	@staticmethod
	def make_maps(shear_files,position_files,npixel=512,smooth=0.5*u.arcmin,fov=3.5*u.deg,zbins=None,add_shape_noise=False,photoz_bias=None,photoz_sigma=None):

		"""
		:param shear_files: list of files that contain the shear catalogs
		:type shear_files: list.

		:param position_files: list of files that contain positions and redshift of the galaxies in the shear files
		:type position_files: list.

		:returns: list of ConvergenceMap

		"""

		#Safety checks
		assert len(shear_files)==len(position_files)

		#Read in all the shear catalogs, and convert to convergence maps with E/B mode decomposition
		convergence_maps = list()

		#Combine single catalog files in a big catalog
		full_catalog = ShearCatalog.readall(shear_files,position_files)

		#Set seed for random noise generation
		try:
			seed = int(real.search(shear_files[0]).groups()[0])
		except AttributeError:
			seed = None
			
		#Generate shape noise and add it to the catalog
		if add_shape_noise:
			shape_noise = full_catalog.shapeNoise(seed)
			full_catalog["shear1"] += shape_noise["shear1"]
			full_catalog["shear2"] += shape_noise["shear2"]

		#Generate photoz errors
		if (photoz_bias is not None) or (photoz_sigma is not None):
			full_catalog["z"] += generate_gaussian_photoz_errors(full_catalog["z"],bias=photoz_bias,sigma=photoz_sigma)

		#Re-bin the catalog to produce tomographic shear maps
		if zbins is not None:
			full_catalog_rebin = full_catalog.rebin(zbins,field="z")
		else:
			full_catalog_rebin = [full_catalog]

		#Append to the redshift binned convergence map
		for sc in full_catalog_rebin:
			convergence_maps.append(sc.toMap(map_size=fov,npixel=npixel,smooth=smooth).convergence())

		#Return
		return convergence_maps

################################
#####FisherDatabase class#######
################################

class FisherDatabase(Database):

	z_bins = [(0.0052829915857917076, 0.46370802037163456), (0.46370802037163456, 0.68921284184762954),(0.68921284184762954, 0.93608623056054063),(0.93608623056054063, 1.2872107430836479),(1.2872107430836479, 2.9998163872653354)]

	#Retrieve the parameter covariance matrix from a row in the database
	def query_parameter_covariance(self,feature_label,nbins=None,table_name="pcov",parameters=["Om","w","sigma8"]):

		"""
		Retrieve the parameter covariance matrix from a row in the database

		:param feature_label: name of the feature to retrieve
		:type feature_label: str.

		:param nbins: number of bins
		:type nbins: int.

		:param table_name: name of the table to query
		:type table_name: str.

		:param parameters: parameter list 
		:type parameters: list.

		:returns: parameter covariance matrix
		:rtype: Ensemble
		
		"""

		#Columns that contain the parameter cross covariances
		pcolumns = ['"{0}-{1}"'.format(i,j) for i,j in product(parameters,parameters)]

		#Build the SQL query
		sql_query = "SELECT {0} ".format(",".join(pcolumns))
		sql_query += "FROM {0} ".format(table_name)
		sql_query += "WHERE feature_label='{0}' ".format(feature_label)
		if nbins is not None:
			sql_query += "AND bins={0}".format(nbins)

		#Perform the query on the database: it should return only one row
		query_row = self.query(sql_query)
		if len(query_row)==0:
			raise ValueError("No record in the database found with these filters!")

		if len(query_row)>1:
			raise ValueError("Multiple records in the database! Something is wrong!")

		#Cast the row into a matrix
		pcov = query_row.values.reshape((len(parameters),)*2)
		return SquareMatrix(pcov,index=parameters,columns=parameters)

	#Retrieve the variance of a single parameter for a single feature_label
	def query_parameter_simple(self,feature_label,table_name="pcov",parameter="w"):

		"""
		Retrieve the variance of a single parameter for a single feature_label

		:param feature_label: name of the feature to retrieve
		:type feature_label: str.

		:param table_name: name of the table to query
		:type table_name: str.

		:param parameter: parameter to retrieve
		:type parameter: str.

		:returns: number of components, parameter variance
		:rtype: tuple.

		"""

		#Build the SQL query and retrieve the results
		sql_query = 'SELECT bins,"{0}" FROM {1} WHERE feature_label="{2}"'.format("-".join([parameter,parameter]),table_name,feature_label)
		query_results = self.query(sql_query)

		#Return the number of components and corresponding parameter variance
		return query_results["bins"].values.astype(np.int),query_results["-".join([parameter,parameter])].values


	#Retrieve the best fit parameters to the data
	def query_parameter_fit(self,feature_label,table_name="pcov",parameters=["Om","w"]):

		"""
		Retrieve the variance of a single parameter for a single feature_label

		:param feature_label: name of the feature to retrieve
		:type feature_label: str.

		:param table_name: name of the table to query
		:type table_name: str.

		:param parameters: parameters to retrieve
		:type parameter: list.

		:returns: parameters best fit 
		:rtype: Ensemble

		"""

		#Build the SQL query and retrieve the results
		sql_query = 'SELECT {0},bins FROM {1} WHERE feature_label="{2}"'.format(",".join(["{0}_fit".format(p) for p in parameters]),table_name,feature_label)
		query_results = self.query(sql_query)

		#Return the query
		return query_results

	######################################################################################################################################################################

	#Plot shortcut by redshift
	def plot_by_redshift(self,features,table_name,parameter,colors):
		fig,axes = plt.subplots(2,3,figsize=(24,16))
		axes = axes.reshape(6)

		#Plot single redshift
		for n in range(5):
			for nfeat,feature in enumerate(features):
				b,p = self.query_parameter_simple(feature+"_z{0}".format(n),table_name,parameter)
				axes[n].plot(b,np.sqrt(p),color=colors[nfeat],label=feature)
				axes[n].set_title(r"$z\in[{0:.2f},{1:.2f}]$".format(*self.__class__.z_bins[n]))

		#Plot alltogether
		for nfeat,feature in enumerate(features):
			b,p = self.query_parameter_simple(feature,table_name,parameter)
			axes[-1].plot(b,np.sqrt(p),color=colors[nfeat],label=feature)
			axes[-1].set_title(r"$z$"+ " tomography")

		#Axes labels
		for ax in axes:
			ax.set_xlabel("PCA components",fontsize=18)
			ax.set_ylabel(r"$\Delta$" + parameter)
			ax.set_yscale("log")
			ax.legend(prop={"size":10})

		#Tight layout
		fig.tight_layout()

		#Return
		return fig,axes

	#Plot shortcut by feature
	def plot_by_feature(self,features,table_name,parameter,title=True):
		fig,axes = plt.subplots(1,len(features),figsize=(8*len(features),8))

		try:
			len(axes)
		except TypeError:
			axes = [axes]

		#Plot alltogether
		for nfeat,feature in enumerate(features):
			b,p = self.query_parameter_simple(feature,table_name,parameter)
			axes[nfeat].plot(b,np.sqrt(p),linewidth=3.5,label=r"$z$"+ r" ${\rm tomography}$")
		
		#Plot single redshift
		for nfeat,feature in enumerate(features):
			for n in range(5):
				b,p = self.query_parameter_simple(feature+"_z{0}".format(n),table_name,parameter)
				axes[nfeat].plot(b,np.sqrt(p),label=r"$z\in[{0:.2f},{1:.2f}]$".format(*self.z_bins[n]))
			axes[nfeat].set_title(r"${\rm " + feature.replace("_pca","").replace("_","\,\,") + r"}$",fontsize=30)

		#Axes labels
		for ax in axes:
			ax.set_xlabel(r"$N_c$",fontsize=22)
			ax.set_ylabel(r"$\Delta$" + parameter,fontsize=22)
			ax.set_yscale("log")
			ax.legend(prop={"size":25})

		#Tight layout
		fig.tight_layout()

		#Return
		return fig,axes


###################
#DESimulationBatch#
###################

class DESimulationBatch(SimulationBatch):

	_parameters = ["Om","Ode","w","wa","si"]
	_fiducial_params = {"Om":0.26,"Ode":0.74,"w":-1.,"si":0.8,"wa":0.}
	_fisher_variations = {"Om":[0.23,0.29],"w":[-1.2,-0.8],"wa":[-0.5,-0.2]}

	@property
	def pformat(self):
		return "{0:."+str(self.environment.cosmo_id_digits)+"f}"

	####################################################################################################################

	@property
	def fiducial_model(self):
		return self.getModel(self.fiducial_cosmo_id)

	@property
	def fisher_variation_models(self):
		for cosmoid in self.fisher_variation_cosmo_id:
			yield self.getModel(cosmoid)

	@property 
	def non_fiducial_models(self):
		return [ m for m in self.models if m.cosmo_id!=self.fiducial_cosmo_id]

	####################################################################################################################

	@property 
	def fiducial_cosmo_id(self):
		return "_".join([p + self.pformat.format(self._fiducial_params[p]) for p in self._parameters])

	@property 
	def fisher_variation_cosmo_id(self):

		#Fiducial
		cosmopar = dict()
		for p in self._parameters:
			cosmopar[p] = self._fiducial_params[p]

		#Variations
		for p in self._parameters:
			
			if p not in self._fisher_variations:
				continue

			for pval in self._fisher_variations[p]:

				if p=="Om":
					
					cosmopar["Om"] = pval
					cosmopar["Ode"] = 1-pval
					yield "_".join([par + self.pformat.format(cosmopar[par]) for par in self._parameters])
					cosmopar["Om"] = self._fiducial_params["Om"]
					cosmopar["Ode"] = self._fiducial_params["Ode"]

				else:

					cosmopar[p] = pval
					yield "_".join([par + self.pformat.format(cosmopar[par]) for par in self._parameters])
					cosmopar[p] = self._fiducial_params[p]

	####################################################################################################################

	@property
	def fiducial_cosmology(self):
		
		cosmopar = dict()
		for p in self._parameters:
			cosmopar[self.environment.name2attr[p]] = self._fiducial_params[p]
		
		return LensToolsCosmology(**cosmopar)

	@property 
	def fisher_variation_cosmology(self):
		for cosmo_id in self.fisher_variation_cosmo_id:
			yield string2cosmo(cosmo_id,name2attr=self.environment.name2attr)[0]


	
