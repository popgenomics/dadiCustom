#! ~/py33/bin
# -*- coding: utf-8 -*-
# ! I changed maxiter from 20 to 1000
# Library importation

import os
import sys
import getopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import pylab
import time
from scipy import stats

import numpy
from numpy import array

import site
site.addsitedir('/home/cchriste/dadi/dadi-1.6.3_modif')
import dadi


import modeledemo_mis_0_d


#Help function
def usage():
	""" Function for help """
	print("# This script allow you to test different demographic models on your genomic data\n"+
	      "# and will give you which one is the best fitted.\n\n"+
	      "# This is an exemple of the most complete command line :\n"+
	      "# -o pathoutput -y population1 -x population2 -p 10,20,30 -f pathfsfile -m SI,IM,IMex,IMex2,AM,PAM,SC,PSC,IM2M2P,AM2M2P,PAM2M2P,SC2M2P,PSC2M2P -l -a -h -v\n\n"+
	      "# This is an exemple of the shortest command line:\n"+
	      "# -f pathfsfile\n\n"+
	      "# -h --help : Display the help you are looking at.\n"+
	      "# -v --verbose : Print steps while the code is running\n"+
	      "# -y --population1 : Take the name of the first population in the sfs (y-axis)\n"+
	      "# -x --population2 : Take the name of the second population in the sfs (x-axis)\n"+
	      "# -o --outputname : Take the path of output file.\n"+
	      "# -f --fs_file_name : Take the path of the fs file from thr parent directory.\n"+
	      "# -p --grid_points : Take 3 numbers separated by a coma, for the size of grids for extrapolation.\n"+
	      "# -m --model_list : Take until 11 names of model (SI,IM,IMex,IMex2,AM,PAM,SC,PSC,IM2M2P,AM2M2P,PAM2M2P,SC2M2P,PSC2M2P) separated by a coma.\n"+
	      "# For more information on models see docstrings in the module modeledemo.\n"+
	      "# -z : mask the singletons.\n"+
	      "# -l : record the final parameters in the output file.\n\n\n"
	      "########################## Enjoy ###########################")
	return()
	      
	      

#Argument function
def takearg(argv):
	""" Function which record arguments from the command line."""
	# default values
	masked = False # freq 0,1 and 1,0 masked if masked = 1
	pts_l = None  # Grids sizes for extrapolation
	outputname = "mis_fs_2d_optlog"
	model_list = ["SI","SIex","SI2N","SI2Nex","IM","IMex","IM2N","IM2Nex","IM2M2P","IM2M2Pex","IM2N2M2P","IM2N2M2Pex","AM","AMex","AM2N","AM2Nex","AM2M2P","AM2M2Pex","AM2N2M2P","AM2N2M2Pex","PAM","PAMex","PAM2N","PAM2Nex","PAM2M2P","PAM2M2Pex","PAM2N2M2P","PAM2N2M2Pex","SC","SCex","SC2N","SC2Nex","SC2M2P","SC2M2Pex","SC2N2M2P","SC2N2M2Pex","PSC","PSCex","PSC2N","PSC2Nex","PSC2M2P","PSC2M2Pex","PSC2N2M2P","PSC2N2M2Pex"]	
	verbose = False
	logparam = False
	nompop1 = "Pop1"
	nompop2 = "Pop2"

	checkfile = False #initilization. if True fs file needed exists, if False it doesn't

	if len(argv) < 2:
		print("You should give, at least, the name of the fs file !")
		sys.exit(1)
	try:
		opts, args = getopt.getopt(argv[1:], "hvo:y:x:azf:p:m:l", ["help", "verbose", "outputname=", "population1=", "population2=", "masked", "fs_file_name=", "grid_points=", "model_list=", "log"])
	except getopt.GetoptError as err:
		# Affiche l'aide et quitte le programme
		print(err) # Va afficher l'erreur en anglais
		usage() # Fonction à écrire rappelant la syntaxe de la commande
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
		else:
			print("Option {} inconnue".format(opt))
			sys.exit(2)
	if not checkfile:
		print("You should give, at least, the name of the fs file !")
		sys.exit(1)
	return(masked, pts_l, outputname, nompop1, nompop2, fs_file_name, model_list, verbose, logparam)


#Inference function
def callmodel(func, data, output_file, output_file_2, modeldemo, ll_opt_dic, nbparam_dic,
	      nompop1="Pop1", nompop2="Pop2", params=None, fixed_params=None, lower_bound=None, upper_bound=None,
	      pts_l=None, ns=None,outputname=None, verbose=False, maxiter=25000, 
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
	line = ("\n" + str(modeldemo) + "\n" + "Model log-likelihood: " + repr(ll_model) + "\n" "Optimization : " + repr(optimizationstate) + "\n"  "Optimized parameters: " + repr(popt) + "\n" + "Optimized log-likelihood: " + repr(ll_opt) + "\n" + "theta: " + repr(theta) + "\n" + "AIC: " + repr(AIC) + "\n")
	output_file.write(line)

	# Plot a comparison of the resulting fs with the data.
        if optimizationstate == "BFGS" :
		import pylab
		pylab.figure()
		dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=1, vmax=1000, resid_range=3,
							    pop_ids =(nompop1,nompop2),
							    saveplot=True, nomplot=(outputname + "_" + modeldemo), showplot=False)
		line = (str(modeldemo) + "\n" + "Optimized log-likelihood: " + "\n"+ repr(ll_opt)+ "\n" + "AIC: " + "\n"+ repr(AIC) + "\n")
		output_file_2.write(line)					    	
 	done=True
	return(done, ll_opt_dic, nbparam_dic, popt)

##############################
##############################

# Load parameters
masked, pts_l, outputname, nompop1, nompop2, fs_file_name, model_list, verbose, logparam = takearg(sys.argv)
	
if pts_l != None:
	for i in range(len(pts_l)):
		pts_l[i] = int(pts_l[i])

# Load the data
data = dadi.Spectrum.from_file(fs_file_name)
ns = data.sample_sizes

# Creation of outputname and setting default params if they are not in the args
datastate = "not_masked"
opt_list = ["anneal_hot", "anneal_cold", "BFGS"]
if pts_l == None:
	pts_l = [ns[0]+10,ns[0]+20,ns[0]+30]
if masked:
	data.mask[1,0] = True
	data.mask[0,1] = True
	outputname = outputname + "_masked"
	datastate = "masked"

outputname = outputname + "_" + repr(time.localtime()[0]) + "_" + repr(time.localtime()[1]) + "_" + repr(time.localtime()[2]) + "_" + repr(time.localtime()[3]) + repr(time.localtime()[4]) + repr(time.localtime()[5])

# Create output dir and file
os.mkdir("../" + outputname)
output_file = open(("../" + outputname + "/" + outputname + ".txt"), "w")

# Save the parameters
if logparam :
	line = ("Model(s) : " + repr(model_list) + "\n" + "Data state : " + repr(datastate) + "\n" + "Grid points : " + repr(pts_l) + "\n\n\n")
	output_file.write(line)
	
output_file_2 = open(("../" + outputname + "/" + outputname + "_2.txt"), "w")	
	
# Create dic for ll to make lrt
ll_opt_dic = {}
nbparam_dic = {}

# ML inference for each model
for namemodel in model_list:
	print namemodel
	time.sleep(1.0)

	if namemodel == "SI":

		# Custom Simple Isolation model: nu1, nu2, Ts, O
		func = modeledemo_mis_0_d.SI

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3])
			else:
				params = (popt[0], popt[1], popt[2], popt[3])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 12, 1.00]
			lower_bound = [0.01, 0.01, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "SIex":

		# Custom Simple Isolation model with expansion after the split: nu1a, nu2a, nu1, nu2, Ts, Te, O
		func = modeledemo_mis_0_d.SIex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "SI2N":

		# Custom Simple Isolation model with different recombination along the genome: nu1, nu2, Ts, nr, bf, O
		func = modeledemo_mis_0_d.SI2N

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 12, 1, 0.999 , 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "SI2Nex":

		# Custom Simple Isolation model with different recombination along the genome: nu1a, nu2a, nu1, nu2, Ts, Te, nr, bf, O
		func = modeledemo_mis_0_d.SI2Nex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else:
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 12, 12, 1, 0.999 , 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if namemodel == "IM":

		# Custom Isolation with Migration model: nu1, nu2, m12, m21, Ts, O
		func = modeledemo_mis_0_d.IM

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		

	if namemodel == "IMex":

		# Custom Isolation with Migration model with expansion: nu1a, nu2a, nu1, nu2, Ts, Te, m12, m21, O
		func = modeledemo_mis_0_d.IMex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (0.1, 0.1, 1, 1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			# really really really not sure of ub and lb parameters for s, it just put the same as nu1 and nu2, but it seems not correct because if expansion ...
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	if namemodel == "IM2N":

		# Custom Isolation with Migration model: nu1, nu2, m12, m21, Ts, nr, bf, O
		func = modeledemo_mis_0_d.IM2N

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7] )
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 1, 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))		
	if namemodel == "IM2Nex":

		# Custom Isolation with Migration model: nu1a, nu2a, nu1, nu2, m12, m21, Ts,Te, nr, bf, O
		func = modeledemo_mis_0_d.IM2Nex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1 , 1 ,1 , 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10] )
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 1, 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	if namemodel == "IM2M2P":

		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, P1, P2, O
		func = modeledemo_mis_0_d.IM2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "IM2M2Pex":

		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P1, P2, O
		func = modeledemo_mis_0_d.IM2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01,0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "IM2N2M2P":

		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, nr, bf, P1, P2, O
		func = modeledemo_mis_0_d.IM2N2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.0, 0.0, 0.0, 0, 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if namemodel == "IM2N2M2Pex":

		# Custom Isolation with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Te, P1, P2, O
		func = modeledemo_mis_0_d.IM2N2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12])
		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01,0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0, 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	
	if (namemodel == "AM") or (namemodel == "PAM"):

		# Custom Ancient Migration Model: nu1, nu2, m12, m21, Tam,Ts, O
		if namemodel == "AM":
 		    func = modeledemo_mis_0_d.AM
                else :
 		    func = modeledemo_mis_0_d.PAM

		for optimizationstate in opt_list:
			print optimizationstate
		
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AMex") or (namemodel == "PAMex"):

		# Custom Ancient Migration Model: nu1a, nu2a, nu1, nu2, m12, m21, Tam, Ts, Te, O
		if namemodel == "AMex":
 		    func = modeledemo_mis_0_d.AMex
                else :
 		    func = modeledemo_mis_0_d.PAMex

		for optimizationstate in opt_list:
			print optimizationstate
		
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6],popt[7],popt[8],popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6],popt[7],popt[8],popt[9])

		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AM2N") or (namemodel == "PAM2N"):

		# Custom Ancient Migration Model: nu1, nu2, m12, m21, Tam, Ts, nr, bf, O
		if namemodel == "AM2N":
 		    func = modeledemo_mis_0_d.AM2N
                else :
 		    func = modeledemo_mis_0_d.PAM2N

		for optimizationstate in opt_list:
			print optimizationstate
		
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12,1 , 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AM2Nex") or (namemodel == "PAM2Nex"):
		# Custom Ancient Migration Model: nu1a, nu2a,nu1, nu2, m12, m21, Tam, Ts, Te, nr, bf, O
		if namemodel == "AM2Nex":
 		    func = modeledemo_mis_0_d.AM2Nex
                else :
 		    func = modeledemo_mis_0_d.PAM2Nex

		for optimizationstate in opt_list:
			print optimizationstate
		
			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12,1 , 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	if (namemodel == "AM2M2P") or (namemodel == "PAM2M2P"):

		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Tam, Ts, P1, P2
		if namemodel == "AM2M2P":
		    func = modeledemo_mis_0_d.AM2M2P
		else :
		    func = modeledemo_mis_0_d.PAM2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AM2M2Pex") or (namemodel == "PAM2M2Pex"):

		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Tam, Ts, Te, P1, P2
		if namemodel == "AM2M2Pex":
		    func = modeledemo_mis_0_d.AM2M2Pex
		else :
		    func = modeledemo_mis_0_d.PAM2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	if (namemodel == "AM2N2M2P") or (namemodel == "PAM2N2M2P"):

		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Tam, Ts, nr, bf, P1, P2, O
		if namemodel == "AM2N2M2P":
		    func = modeledemo_mis_0_d.AM2N2M2P
		else :
		    func = modeledemo_mis_0_d.PAM2N2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0, 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "AM2N2M2Pex") or (namemodel == "PAM2N2M2Pex"):

		# Custom Ancient Migration with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Tam, Ts, Te, nr, bf, P1, P2, O
		if namemodel == "AM2N2M2Pex":
		    func = modeledemo_mis_0_d.AM2N2M2Pex
		else :
		    func = modeledemo_mis_0_d.PAM2N2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = ( 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12, 1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0 , 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "SC") or (namemodel == "PSC"):

		# Custom Simple Secondary Contact Model: nu1, nu2, m12, m21, Ts, Tsc, O
		if namemodel == "SC":
 		    func = modeledemo_mis_0_d.SC
                else :
 		    func = modeledemo_mis_0_d.PSC
		
		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6])

		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
		
	if (namemodel == "SCex") or (namemodel == "PSCex"):

		# Custom Simple Secondary Contact Model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, O
		if namemodel == "SCex":
 		    func = modeledemo_mis_0_d.SCex
                else :
 		    func = modeledemo_mis_0_d.PSCex
		
		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9])

		
			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100,100, 100, 100, 20, 20, 12, 12, 12, 1.00]
			lower_bound = [0.01,0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "SC2N") or (namemodel == "PSC2N"):

		# Custom Simple Secondary Contact Model: nu1, nu2, m12, m21, Ts, Tsc, nr, bf, O
		if namemodel == "SC2N":
 		    func = modeledemo_mis_0_d.SC2N
                else :
 		    func = modeledemo_mis_0_d.PSC2N
		
		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1, 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

		# Custom Simple Secondary Contact Model: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te nr, bf, O
		
	if (namemodel == "SC2Nex") or (namemodel == "PSC2Nex"):
				
		if namemodel == "SC2Nex":
 		    func = modeledemo_mis_0_d.SC2Nex
                else :
 		    func = modeledemo_mis_0_d.PSC2Nex
		
		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12, 1, 0.999, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.001, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file,output_file_2, namemodel, ll_opt_dic, nbparam_dic,
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
	if (namemodel == "SC2M2P") or (namemodel == "PSC2M2P"):

		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tsc, P1, P2
		if namemodel == "SC2M2P":
		    func = modeledemo_mis_0_d.SC2M2P
		else :
		    func = modeledemo_mis_0_d.PSC2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "SC2M2Pex") or (namemodel == "PSC2M2Pex"):

		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, P1, P2
		if namemodel == "SC2M2Pex":
		    func = modeledemo_mis_0_d.SC2M2Pex
		else :
		    func = modeledemo_mis_0_d.PSC2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
		
	if (namemodel == "SC2N2M2P") or (namemodel == "PSC2N2M2P"):

		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1, nu2, m12, m21, Ts, Tsc, nr, bf, P1, P2, O
		if namemodel == "SC2N2M2P":
		    func = modeledemo_mis_0_d.SC2N2M2P
		else :
		    func = modeledemo_mis_0_d.PSC2N2M2P

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 20, 20, 12, 12, 1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0, 0, 0, 0, 0, 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))

	if (namemodel == "SC2N2M2Pex") or (namemodel == "PSC2N2M2Pex"):

		# Custom Secondary contact with 2 Migration rate model and 2 Proportion of loci: nu1a, nu2a, nu1, nu2, m12, m21, Ts, Tsc, Te, nr, bf, P1, P2
		if namemodel == "SC2N2M2Pex":
		    func = modeledemo_mis_0_d.SC2N2M2Pex
		else :
		    func = modeledemo_mis_0_d.PSC2N2M2Pex

		for optimizationstate in opt_list:
			print optimizationstate

			if optimizationstate == "anneal_hot":		
				params = (1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5, 1, 0.5, 0.5, 0.8)
			elif optimizationstate == "anneal_cold":
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13])
			else :
				params = (popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13])

			# The upper_bound array is for use in optimization. Occasionally the optimizer
			# will try wacky parameter values. We in particular want to exclude values with
			# very long times, as they will take a long time to evaluate.
			upper_bound = [100, 100, 100, 100, 20, 20, 12, 12, 12,  1, 0.999, 1.00, 1.00, 1.00]
			lower_bound = [0.01, 0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0.001, 0.00, 0.00, 0.00]

			done, ll_opt_dic, nbparam_dic, popt = callmodel(func, data, output_file, output_file_2, namemodel, ll_opt_dic, nbparam_dic, 
								  nompop1=nompop1, nompop2=nompop2, params=params, fixed_params=None, lower_bound=lower_bound, 
								  upper_bound=upper_bound,  pts_l=pts_l, ns=ns,
								  outputname="../" + outputname + "/" + outputname, 
								  verbose=verbose, maxiter=25000, Tini=50, Tfin=0, learn_rate=0.005, 
								  schedule= "cauchy")
		if done: print(("\n" + namemodel + " : done\n"))
output_file.close()
output_file_2.close()
