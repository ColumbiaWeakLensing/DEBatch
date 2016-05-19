#!/usr/bin/env python

import os
import sys
import argparse

#Don't need MPI here
sys.modules["mpi4py"] = None

from lenstools import data as lensData
from lenstools import SimulationBatch
from lenstools.pipeline.settings import *
from lenstools.pipeline.cluster import *
from lenstools.pipeline.deploy import ParsedHandler

#Dictionary that converts command line argument into the appropriate job handler
system2handler = {
"Stampede" : StampedeHandler,
"stampede" : StampedeHandler,
"Edison" : EdisonHandler,
"edison" : EdisonHandler,
"Cori" : CoriHandler,
"cori" : CoriHandler,
}

#Parse command line options
parser = argparse.ArgumentParser()
parser.add_argument("-e","--environment",dest="env_file",action="store",type=str,default=lensData("environment_default.ini"),help="environment option file")
parser.add_argument("-o","--options",dest="exec_options",action="store",type=str,default=lensData("lens_default.ini"),help="configuration file to pass to the executable (planes or raytracing)")
parser.add_argument("-j","--job",dest="job_options_file",action="store",type=str,default=lensData("job_default.ini"),help="job specifications file")
parser.add_argument("-s","--system",dest="system",action="store",type=str,default=None,help="configuration file that contains the cluster specifications")
parser.add_argument("-i","--input",dest="input",action="store",type=str,default="camb.params",help="name of the camb parameter file")
parser.add_argument("-l","--log",dest="log",action="store",type=str,default="camb_linear.out",help="name of the camb output log file")
parser.add_argument("model_file",nargs="?",default=None,help="text file that contains all the IDs of the models to include in the job submission")

#Parse command arguments and check that all provided options are available
cmd_args = parser.parse_args()

#Log to user
print("[*] Environment settings for current batch read from {0}".format(cmd_args.env_file))
environment = EnvironmentSettings.read(cmd_args.env_file)

#Instantiate handler
if cmd_args.system is not None:
	system = cmd_args.system
else:
	print("[*] No system option provided, reading system type from $THIS environment variable")
	system = os.getenv("THIS")
	if system=="":
		print("[-] $THIS environment variable is not set, cannot continue")
		sys.exit(1)

if system in system2handler:
	print("[+] Using job handler for system {0}".format(system))
	job_handler = system2handler[system]()
else:
	print("[+] Using job handler parsed from {0}".format(system))
	job_handler = ParsedHandler.read(system)


print("[*] Current batch home directory: {0}".format(environment.home))
print("[*] Current batch mass storage: {0}".format(environment.storage))

#Instantiate the simulation batch
batch = SimulationBatch(environment)

#Read the realizations to include in the job submission (if no file is provided read from stdin)
if cmd_args.model_file is not None:
	print("[*] Realizations to include in this submission will be read from {0}".format(cmd_args.model_file))
	with open(cmd_args.model_file,"r") as modelfile:
		realizations = [ l.strip("\n") for l in modelfile.readlines() if l!="" ]
else:
	print("[*] Realizations to include in this submission will be read from stdin")
	realizations = [ l.strip("\n") for l in sys.stdin.readlines() if l!="" ]

#Log to user
print("[+] Found {0} realizations to include in job submission".format(len(realizations)))

#Now decide which type of submission are we generating, and generate the script
job_type = "camb"

if job_type=="camb":

	section = "CAMB"
	print("[+] Generating CAMB submission script")
	print("[*] Reading job specifications from {0} section {1}".format(cmd_args.job_options_file,section))
	job_settings = JobSettings.read(cmd_args.job_options_file,section)
	job_settings.num_cores = len(realizations)
	
	with open(os.path.join(batch.home,"Jobs",job_settings.job_script_file),"w") as fp:
		
		#Preamble
		print("[+] Writing preamble")
		print("[+] CAMB parameters will be read from cosmo_id/geometry_id/{0}".format(cmd_args.input))
		print("[+] CAMB output will be directed to cosmo_id/geometry_id/{0}".format(cmd_args.log))
		fp.write(job_handler.writePreamble(job_settings))
		fp.write("\n"+"##################################\n"*3 + "\n\n")

		#Execution
		for n,r in enumerate(realizations):
			cosmo_id,geometry_id = r.split("|")
			print("[+] Writing executable for cosmology {0}".format(cosmo_id))
			executable = "{0} -n 1 -o {1} {2}".format(job_handler._cluster_specs.job_starter,n,job_settings.path_to_executable)
			executable += " " + os.path.join(batch.getModel(cosmo_id).getCollection(geometry_id).home,cmd_args.input)
			executable += " > {0} &".format(os.path.join(batch.getModel(cosmo_id).getCollection(geometry_id).home,cmd_args.log)) 
			fp.write(executable+"\n\n")

		#Wait
		fp.write("wait\n")

else:
	raise ValueError("This script generates camb submissions only!")








