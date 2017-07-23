#!/usr/bin/python
##! ~/py33/bin
# -*- coding: utf-8 -*-

# CRoux version
# Heliconius


import os
import sys
#sys.path.insert(1, '/home/croux/Documents/local_python/lib/python2.7/site-packages/')
#import pkg_resources


#pkg_resources.require("matplotlib==2.0.1")
#pkg_resources.require("scipy==0.9.0")
#pkg_resources.require("numpy==1.12.1")


import matplotlib
matplotlib.use('Agg')

# Library importation
import dadi
import getopt
import pylab
import time
from scipy import stats

import numpy
from numpy import array

import modeledemo_0
#import modeles

#Help function
def usage():
	""" Function for help """
	print("# This script allow you to test different demographic models on your genomic data\n"+
	      "# and will give you which one is the best fitted.\n\n"+
	      "# This is an exemple of the most complete command line :\n"+
	      "# -o pathoutput -y population1 -x population2 -p 10,20,30 -f pathfsfile -m SI,SI2N,IM,AM,PAM,SC,PSC,IM2M,AM2M,PAM2M,SC2M,PSC2M,IM2M2P,AM2M2P,PAM2M2P,SC2M2P,PSC2M2P -l -a -h -v\n\n"+
	      "# This is an exemple of the shortest command line:\n"+
	      "# -f pathfsfile\n\n"+
	      "# -h --help : Display the help you are looking at.\n"+
	      "# -v --verbose : Print steps while the code is running\n"+
	      "# -y --population1 : Take the name of the first population in the sfs (y-axis)\n"+
	      "# -x --population2 : Take the name of the second population in the sfs (x-axis)\n"+
	      "# -o --outputname : Take the path of output file.\n"+
	      "# -f --fs_file_name : Take the path of the fs file from thr parent directory.\n"+
	      "# -p --grid_points : Take 3 numbers separated by a coma, for the size of grids for extrapolation.\n"+
	      "# -m --model_list : Take until 34 names of model (PAN, PAN2N, PANex, PAN2Nex, SI,SIex,SI2N,SI2Nex,IM,IMex,AM,AMex,PAM,PAMex,SC,SCex,PSC,PSCex,IM2M,IM2Mex,AM2M,AM2Mex,PAM2M,PAM2Mex,SC2M,SC2Mex,PSC2M,PSC2Mex,IM2M2P,IM2M2Pex,AM2M2P,AM2M2Pex,PAM2M2P,PAM2M2Pex,SC2M2P,SC2M2Pex,PSC2M2P,PSC2M2Pex) separated by a coma.\n"+
	      "# For more information on models see docstrings in the module modeledemo.\n"+
	      "# -z : mask the singletons.\n"+
	      "# -r : residual_range (maximum number of SNPs for the color-gradient in residual plots)\n"+
	      "# -c : actives the anneal_cold. If not specified: only do hot + BFGS to win some time.\n"+
	      "# -l : record the final parameters in the output file.\n\n\n"+
	      "# Exemple: ./dadi_folded_expansion.py -o SI_8 -y ros -x chi -f folded_spectrum_all_SNPs.fs -m SI -z -l -n 20000 -r 50 -p 15,25,35 -c\n\n\n"
	      "########################## Enjoy ###########################")
	return()
	      
	

#Argument function
def takearg(argv):
	""" Function which record arguments from the command line."""
	# default values
	masked = False # freq 0,1 and 1,0 masked if masked = 1
	pts_l = None  # Grids sizes for extrapolation
	outputname = "fs_2d_optlog"
	model_list = ["PAN", "PAN2N", "PANex", "PAN2Nex", "SI", "SIex", "SI2N", "SI2Nex", "IM", "IMex", "AM", "AMex", "PAM", "PAMex", "SC", "SCex", "PSC", "PSCex", "IM2M", "IM2Mex", "AM2M", "AM2Mex", "PAM2M", "PAM2Mex", "SC2M", "SC2Mex", "PSC2M", "PSC2Mex", "IM2M2P", "IM2M2Pex", "AM2M2P", "AM2M2Pex", "PAM2M2P", "PAM2M2Pex", "SC2M2P", "SC2M2Pex", "PSC2M2P", "PSC2M2Pex"]
	verbose = False
	logparam = False
	cold = False
	nompop1 = "Pop1"
	nompop2 = "Pop2"
	checkfile = False #initilization. if True fs file needed exists, if False it doesn't

	residRange=30

	if len(argv) < 2:
		usage() # Fonction a ecrire rappelant la syntaxe de la commande
		sys.exit(1)
	try:
		opts, args = getopt.getopt(argv[1:], "hvo:y:x:azf:p:m:lr:c", ["help", "verbose", "outputname=", "population1=", "population2=", "masked", "fs_file_name=", "grid_points=", "model_list=", "log", "residRange=", "cold"])
	except getopt.GetoptError as err:
		# Affiche l'aide et quitte le programme
		print(err) # Va afficher l'erreur en anglais
		usage() # Fonction a ecrire rappelant la syntaxe de la commande
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			usage()                     
			sys.exit()
		elif opt in ("-v", "--verbose"):
			verbose = True
		elif opt in ("-o", "--outputname"):
			outputname = arg
		elif opt in ("-y", "--population1"):
			nompop1 = arg
		elif opt in ("-x", "--population2"):
			nompop2 = arg
		elif opt in ("-z", "--masked"):
			masked = True
		elif opt in ("-f", "--fs_file_name"):
			fs_file_name = arg
			checkfile = True
		elif opt in ("-p", "--grid_points"):
			pts_l = arg.split(",")
		elif opt in ("-m", "--model_list"):
			model_list = arg.split(",")
		elif opt in ("-l", "--log"):
			logparam = True
		elif opt in ("-r", "--residRange"):
			residRange = int(arg)
		elif opt in ("-c", "--cold"):
			cold = True
		else:
			print("Option {} inconnue".format(opt))
			sys.exit(2)
	if not checkfile:
		print("You should give, at least, the name of the fs file !")
		sys.exit(1)
	return(masked, pts_l, outputname, nompop1, nompop2, fs_file_name, model_list, verbose, logparam, residRange, cold)


#Inference function
def callmodel(func, data, output_file, output_file_2, modeldemo, ll_opt_dic, nbparam_dic,
	      nompop1="Pop1", nompop2="Pop2", params=None, fixed_params=None, lower_bound=None, upper_bound=None,
	      pts_l=None, ns=None,outputname=None, outputname2=None, verbose=False, maxiter=3, 
	      Tini=50, Tfin=0, learn_rate=0.005, schedule= "cauchy"):

	# Make the extrapolating version of our demographic model function.
	func_ex = dadi.Numerics.make_extrap_log_func(func)
	# Calculate the model AFS.
	model = func_ex(params, ns, pts_l)
	# Likelihood of the data given the model AFS.
	ll_model = dadi.Inference.ll_multinom(model, data)
	print 'Model log-likelihood:', ll_model
	# The optimal value of theta (4*No*u) given the model.
	theta = dadi.Inference.optimal_sfs_scaling(model, data)
	print 'theta:', theta
	
	# Do the optimization. By default we assume that theta is a free parameter,
	# since it's trivial to find given the other parameters. If you want to fix
	# theta, add a multinom=False to the call.
	# (This is commented out by default, since it takes several minutes.)
	# The maxiter argument restricts how long the optimizer will run. For production
	# runs, you may want to set this value higher, to encourage better convergence.
	# Tini = initial temperature of the chain.
        # Learn rate = decreasing rate in the probability of accepting worse solutions as it explores the solution space. 
	if optimizationstate == "anneal_hot" :
		# Perturb our parameter array before optimization. This does so by taking each
		# parameter a up to a factor of two up or down.
		p0 = dadi.Misc.perturb_params(params, fold=1, lower_bound=lower_bound, upper_bound=upper_bound)
		print 'Starting parameters', repr(p0)
		
		popt = dadi.Inference.optimize_anneal(p0, data, func_ex, pts_l, 
						      lower_bound=lower_bound,
						      upper_bound=upper_bound,
						      verbose=verbose,
						      maxiter=maxiter, Tini=Tini, Tfin=Tfin, 
						      learn_rate=learn_rate, schedule=schedule)
	elif optimizationstate == "anneal_cold" :
		popt = dadi.Inference.optimize_anneal(params, data, func_ex, pts_l, 
						      lower_bound=lower_bound,
						      upper_bound=upper_bound,
						      verbose=verbose,
						      maxiter=maxiter/2, Tini=Tini/2, Tfin=Tfin, 
						      learn_rate=learn_rate*2, schedule=schedule)
 
	else :
		popt = dadi.Inference.optimize_log(params, data, func_ex, pts_l, 
						   lower_bound=lower_bound,
						   upper_bound=upper_bound,
						   verbose=verbose,
						   maxiter=maxiter/2)
	
	# Computation of statistics
	model = func_ex(popt, ns, pts_l)
	ll_opt = dadi.Inference.ll_multinom(model, data)
	theta = dadi.Inference.optimal_sfs_scaling(model, data)
	AIC = 2*len(params)-2*ll_opt
        ll_opt_dic[modeldemo] = ll_opt
        nbparam_dic[modeldemo] = len(params)
	
	# Print results
	print 'Optimized parameters', repr(popt)
	print 'Optimized log-likelihood:', ll_opt
	print 'theta:', theta
	
	# Write results
	line = ("\n" + str(modeldemo) + "\n" + "Model log-likelihood: " + repr(ll_model) + "\n" "Optimization : " + repr(optimizationstate) + "\n"  "Optimized parameters: " + "\t".join([ str(param) for param in popt]) + "\n" + "Optimized log-likelihood: " + repr(ll_opt) + "\n" + "theta: " + repr(theta) + "\n" + "AIC: " + repr(AIC) + "\n")
	output_file.write(line)

	# Plot a comparison of the resulting fs with the data.
        if optimizationstate == "BFGS" :
		import pylab
		pylab.figure()
		#dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=1, vmax=100000, resid_range=100,
		dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=vMin, vmax=vMax, resid_range=residRange, # croux 11/07/2017
							    pop_ids =(nompop1,nompop2),
							    saveplot=True, nomplot=(outputname + "_" + modeldemo), showplot=False)
		line = (str(outputname2) + "\t" + modeldemo + "\t" + repr(ll_opt)+ "\t" + repr(AIC) + "\n")
		output_file_2.write(line)					    	
 	done=True
	return(done, ll_opt_dic, nbparam_dic, popt)

##############################
##############################

# Load parameters
masked, pts_l, outputname, nompop1, nompop2, fs_file_name, model_list, verbose, logparam, residRange, cold = takearg(sys.argv)
residRange = int(residRange)

if pts_l != None:
	for i in range(len(pts_l)):
		pts_l[i] = int(pts_l[i])

# Load the data
data = dadi.Spectrum.from_file(fs_file_name)
ns = data.sample_sizes

# TMP for figure param
tmp = data.copy()
tmp[1][0]=data.min()
tmp[0][1]=data.min()
vMin = int(0.8*numpy.floor(tmp.min()))
vMax = int(1.2*numpy.ceil(tmp.max()))


# Creation of outputname and setting default params if they are not in the args
datastate = "not_masked"
if cold == True:
	opt_list = ["anneal_hot", "anneal_cold", "BFGS"] # CRoux 11/07/2017
else:
	opt_list = ["anneal_hot", "BFGS"]

if pts_l == None:
	pts_l = [ns[0]+10,ns[0]+20,ns[0]+30]
if masked:
	data.mask[1,0] = True
	data.mask[0,1] = True
	outputname = outputname + "_masked"
	datastate = "masked"

outputname = outputname + "_" + repr(time.localtime()[0]) + "_" + repr(time.localtime()[1]) + "_" + repr(time.localtime()[2]) + "_" + repr(time.localtime()[3]) + repr(time.localtime()[4]) + repr(time.localtime()[5])

# Create output dir and file
os.mkdir(outputname)
output_file = open((outputname + "/" + outputname + ".txt"), "w")

# Save the parameters
if logparam :
	line = ("Model(s) : " + repr(model_list) + "\n" + "Data state : " + repr(datastate) + "\n" + "Grid points : " + repr(pts_l) + "\n\n\n")
	output_file.write(line)
	
output_file_2 = open((outputname + "/" + outputname + "_2.txt"), "w")	
	
# Create dic for ll to make lrt
ll_opt_dic = {}
nbparam_dic = {}

# upper bound for prior parameters
nu1_max = 50
nu2_max = 50
nu1a_max = 50
nu2a_max = 50
Ts_max = 20
Te_max = 20
Tam_max = 20
Tsc_max = 20
nr_max = 1
bf_max = 0.999 # bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
m12_max = 10
m21_max = 10
P_max = 1.00
P1_max = 1.00
P2_max = 1.00

# lower bound ofr prior parameters
nu1_min = 0.01
nu2_min = 0.01
nu1a_min = 0.01
nu2a_min = 0.01
Ts_min = 0
Te_min = 0
Tam_min = 0
Tsc_min = 0
nr_min = 0
bf_min = 0.001 # bf : Background factor (to which extent the effective population size is reduced in the non-recombining regions)
m12_min = 0
m21_min = 0
P_min = 0.00
P1_min = 0.00
P2_min = 0.00


# ML inference for each model
for namemodel in model_list:
	print namemodel
	time.sleep(1.0)
	if namemodel == "PAN":
		# Panmixia: nu1, Ts. Needs more than one param to avoid bug. Ts is estimated in a narrow range close to zero
		func = modeledemo_0.PAN
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 0)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1])
			else:
				params = (popt[0], popt[1])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [nu1_max, 0.0001] # CRoux
			lower_bound = [nu1_min, 0]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "PAN2N":
		# Panmixia + hetero_N: nu1, nr, bf
		func = modeledemo_0.PAN2N
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2])
			else:
				params = (popt[0], popt[1], popt[2])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12] # original
			upper_bound = [nu1_max, nr_max, bf_max] # CRoux
			lower_bound = [nu1_min, nr_min, bf_min]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "PANex":
		# Panmixia + expansion: nu1a, nu1, Te
		func = modeledemo_0.PANex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2])
			else:
				params = (popt[0], popt[1], popt[2])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12] # original
			upper_bound = [nu1a_max, nu1_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu1_min, Te_min]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "PAN2Nex":
		# Panmixia + expansion + 2N: nu1a, nu1, nr, bf, Te = params
		func = modeledemo_0.PAN2Nex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 1, 0.5, 0.5, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12] # original
			upper_bound = [nu1a_max, nu1_max, nr_max, bf_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu1_min, nr_min, bf_min, Te_min]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "SI":
		# Custom Simple Isolation model: nu1, nu2, Ts
		func = modeledemo_0.SI
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2])
			else:
				params = (popt[0], popt[1], popt[2])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12] # original
			upper_bound = [nu1_max, nu2_max, Ts_max] # CRoux
			lower_bound = [nu1_min, nu2_min, Ts_min]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "SIex":
		# Custom Simple Isolation model: nu1a, nu2a, nu1, nu2, Ts, Te
		func = modeledemo_0.SIex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, Ts_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, Ts_min, Te_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "SI2N":
		# Custom Simple Isolation model with different recombination along the genome: nu1, nu2, Ts, nr, bf
		func = modeledemo_0.SI2N
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 0.5, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12, 1, 0.999] # original
			upper_bound = [nu1_max, nu2_max, Ts_max, nr_max, bf_max] # CRoux
			lower_bound = [nu1_min, nu2_min, Ts_min, nr_min, bf_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "SI2Nex":
		# Custom Simple Isolation model with different recombination along the genome: nu1a, nu2a, nu1, nu2, Ts, Te, nr, bf
		func = modeledemo_0.SI2Nex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 0.5, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 12, 1, 0.999] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, Ts_max, Te_max, nr_max, bf_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, Ts_min, Te_min, nr_min, bf_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))


	if namemodel == "IM":
		# Custom Isolation with Migration model: nu1, nu2, m12, m21, Ts
		func = modeledemo_0.IM
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "IMex":
		# Custom Isolation with Migration model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te
		func = modeledemo_0.IMex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Te_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AM") or (namemodel == "PAM"):
		# Custom Ancient Migration Model: nu1, nu2, m12, m21, Ts, Tam
		if namemodel == "AM":
 		    func = modeledemo_0.AM
                else :
 		    func = modeledemo_0.PAM
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "AMex") or (namemodel == "PAMex"):
		# Custom Ancient Migration Model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te
		if namemodel == "AMex":
 		    func = modeledemo_0.AMex
                else :
 		    func = modeledemo_0.PAMex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tam_min, Te_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if (namemodel == "SC") or (namemodel == "PSC"):
		# Custom Simple Secondary Contact Model: nu1, nu2, m12, m21, Ts, Tsc
		if namemodel == "SC":
 		    func = modeledemo_0.SC
                else :
 		    func = modeledemo_0.PSC
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "SCex") or (namemodel == "PSCex"):
		# Custom Simple Secondary Contact Model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te
		if namemodel == "SCex":
 		    func = modeledemo_0.SCex
                else :
 		    func = modeledemo_0.PSCex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max, Te_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tsc_min, Te_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if namemodel == "IM2M":
		# Custom Isolation with 2 Migration rate model: nu1, nu2, m12, m21, Ts, P
		func = modeledemo_0.IM2M
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, P_max] # CRoux
			lower_bound = [0.01, 0.01, 0.0, 0.0, 0.0, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "IM2Mex":
		# Custom Isolation with 2 Migration rate model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P
		func = modeledemo_0.IM2Mex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Te_max, P_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Te_min, P_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "AM2M") or (namemodel == "PAM2M"):
		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tam, P
		if namemodel == "AM2M":
			func = modeledemo_0.AM2M
		else :
			func = modeledemo_0.PAM2M
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max, P_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "AM2Mex") or (namemodel == "PAM2Mex"):
		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2am nu1, nu2, m12, m21, Ts, Tam, Te, P
		if namemodel == "AM2Mex":
			func = modeledemo_0.AM2Mex
		else :
			func = modeledemo_0.PAM2Mex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max, Te_max,  P_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tam_min, Te_min,  P_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	
	if (namemodel == "SC2M") or (namemodel == "PSC2M"):
		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tsc, P
		if namemodel == "SC2M":
		    func = modeledemo_0.SC2M
		else :
		    func = modeledemo_0.PSC2M
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max, P_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "SC2Mex") or (namemodel == "PSC2Mex"):
		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P
		if namemodel == "SC2Mex":
		    func = modeledemo_0.SC2Mex
		else :
		    func = modeledemo_0.PSC2Mex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max, Te_max, P_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tsc_min, Te_min, P_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	
	if namemodel == "IM2M2P":
		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, P1, P2
		func = modeledemo_0.IM2M2P
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 1.00, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, P1_max, P2_max] # CRoux
			lower_bound = [0.01, 0.01, 0.0, 0.0, 0.0, 0.00, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "IM2M2Pex":
		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P1, P2
		func = modeledemo_0.IM2M2Pex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 1.00, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Te_max, P1_max, P2_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Te_min, P1_min, P2_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))


	if (namemodel == "AM2M2P") or (namemodel == "PAM2M2P"):
		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tam, P1, P2
		if namemodel == "AM2M2P":
		    func = modeledemo_0.AM2M2P
		else :
		    func = modeledemo_0.PAM2M2P
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max, P1_max, P2_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "AM2M2Pex") or (namemodel == "PAM2M2Pex"):
		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tam, Te, P1, P2
		if namemodel == "AM2M2Pex":
		    func = modeledemo_0.AM2M2Pex
		else :
		    func = modeledemo_0.PAM2M2Pex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tam_max, Te_max, P1_max, P2_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tam_min, Te_min, P1_min, P2_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	
	if (namemodel == "SC2M2P") or (namemodel == "PSC2M2P"):
		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tsc, P1, P2
		if namemodel == "SC2M2P":
		    func = modeledemo_0.SC2M2P
		else :
		    func = modeledemo_0.PSC2M2P
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00] # original
			upper_bound = [nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max, P1_max, P2_max] # CRoux
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00, 0.00]
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	
	if (namemodel == "SC2M2Pex") or (namemodel == "PSC2M2Pex"):
		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P1, P2
		if namemodel == "SC2M2Pex":
		    func = modeledemo_0.SC2M2Pex
		else :
		    func = modeledemo_0.PSC2M2Pex
		for optimizationstate in opt_list:
			print optimizationstate
			if optimizationstate == "anneal_hot":		
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			#upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00] # original
			upper_bound = [nu1a_max, nu2a_max, nu1_max, nu2_max, m12_max, m21_max, Ts_max, Tsc_max, Te_max, P1_max, P2_max] # CRoux
			lower_bound = [nu1a_min, nu2a_min, nu1_min, nu2_min, m12_min, m21_min, Ts_min, Tsc_min, Te_min, P1_min, P2_min] # CRoux
			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname= outputname + "/" + outputname,  outputname2=outputname,
								  verbose=verbose, maxiter=20, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
output_file.close()
output_file_2.close()
