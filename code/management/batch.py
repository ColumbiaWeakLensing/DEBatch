from lenstools import SimulationBatch

###################
#DESimulationBatch#
###################

class DESimulationBatch(SimulationBatch):

	_fiducial_params = {"Om":0.26,"Ode":0.74,"w":-1.,"si":0.8,"wa":0.}
	_parameters = ["Om","Ode","w","wa","si"]

	@property
	def fiducial_model(self):
		return self.getModel(self.fiducial_cosmo_id)

	@property
	def non_fiducial_models(self):
		return [ m for m in self.models if m.cosmo_id!=self.fiducial_cosmo_id ]

	@property
	def pformat(self):
		return "{0:."+str(self.environment.cosmo_id_digits)+"f}"

	@property 
	def fiducial_cosmo_id(self):
		return "_".join([p + self.pformat.format(self._fiducial_params[p]) for p in self._parameters])