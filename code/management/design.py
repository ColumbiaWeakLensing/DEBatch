#!/usr/bin/env python

import sys
import json
import argparse
import itertools
import matplotlib.pyplot as plt

sys.modules["mpi4py"] = None

from lenstools.simulations.design import Design

name2label = {"Om" : r"$\Omega_m$", "w0" : r"$w_0$" , "wa" : r"$w_a$" , "sigma8" : r"$\sigma_8$"}

def main():

	#Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("parfile",nargs=1)
	parser.add_argument("-n","--npoints",dest="npoints",action="store",type=int,default=50,help="Number of points in the design")
	parser.add_argument("-m","--maxiter",dest="maxiter",action="store",type=int,default=10000,help="Number of iterations for the sampling")
	parser.add_argument("-v","--visualize",dest="visualize",action="store",type=str,default=None,help="Shape of the panel array to visualize")
	cmd_args = parser.parse_args()

	#Read in the file with parameter specifications
	with open(cmd_args.parfile[0]) as fp:
		par = json.load(fp)

	#Instantiate the design and sample
 	parspecs = [(p,name2label[p],par[p][0],par[p][1]) for p in par]
	design = Design.from_specs(cmd_args.npoints,parspecs)
	design.sample(maxIterations=cmd_args.maxiter)

	#Save
	design.save(cmd_args.parfile[0].replace(".json",".pkl"))

	#Maybe visualize
	if cmd_args.visualize is not None:

		r,c = [ int(n) for n in cmd_args.visualize.split(",") ]
		fig,ax = plt.subplots(r,c,figsize=(8*c,8*r))
		ax = ax.reshape(r*c)

		for n,p in enumerate(itertools.combinations(design,2)):
			design.visualize(fig=fig,ax=ax[n],parameters=p,marker="x",s=50)

		fig.savefig(cmd_args.parfile[0].replace(".json",".png"))


######################################################

if __name__=="__main__":
	main()