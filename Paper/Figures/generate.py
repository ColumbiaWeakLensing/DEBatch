#!/usr/bin/env python
from __future__ import division

import sys,os,argparse
sys.modules["mpi4py"] = None

import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lenstools.pipeline.simulation import SimulationBatch
from lenstools.statistics.constraints import FisherAnalysis

#Options
parser = argparse.ArgumentParser()
parser.add_argument("-t","--type",dest="type",default="png",help="format of the figure to save")
parser.add_argument("fig",nargs="*")

#########################################
#Global: SimulationBatch handle + models#
#########################################

batch = SimulationBatch.current()
models = batch.models

fiducial = models[4]
variations1 = [ models[n] for n in [6,1,2] ]
variations2 = [ models[n] for n in [0,5,3] ]

###################################################################################################
###################################################################################################

def pb2Bias(cmd_args,fontsize=22):
	
	fig,ax = plt.subplots()

	ell = np.load(os.path.join(batch.home,"ell_nb98.npy"))
	p = np.load(os.path.join(fiducial["c0M0"].home,"power_s0_nb98.npy"))

	ax.plot(ell,p.mean(0))

	fig.savefig("pb2Bias."+cmd_args.type)

###################################################################################################
###################################################################################################
###################################################################################################

#Method dictionary
method = dict()
method["1"] = pb2Bias

#Main
def main():
	cmd_args = parser.parse_args()
	for fig in cmd_args.fig:
		method[fig](cmd_args)

if __name__=="__main__":
	main()