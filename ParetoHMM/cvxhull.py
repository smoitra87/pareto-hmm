"""
Contains all the routines for finding the convex hull/ pareto frontier

@author:smoitra@cs.cmu.edu
"""

import numpy as np
import os,sys
import pylab as pl
from HMM import HMM,CMRF, TMRF
from pdb import set_trace as stop
import warnings

def pareto_frontier(cmrf,featlist) :
	"""Finds and prints the pareto frontier. Currently works for 
		two competing states only
	 """
	Q = []
	nStates = len(featlist)
	feat1,feat2 = featlist
	Eaxa,Xa = cmrf.decode(feat1)
	Ebxb,Xb = cmrf.decode(feat2)
	if Xa == Xb : 
		return [Xa],[(Eaxa,Ebxb)]
	Eaxb = cmrf.score(Xb,feat1)
	Ebxa = cmrf.score(Xa,feat2)
	Q.append((Xa,Xb))
	frontier,frontier_energy = [],[]
	frontier.extend([Xa,Xb])
	frontier_energy.extend([(Eaxa,Ebxa),(Eaxb,Ebxb)])
	while len(Q) > 0 :
		### Optimize 
		Xa,Xb = Q[0]
		Q = Q[1:] # Dequeue
		Eaxb = cmrf.score(Xb,feat1)
		Ebxa = cmrf.score(Xa,feat2)	
		Eaxa = cmrf.score(Xa,feat1)
		Ebxb = cmrf.score(Xb,feat2)	
		m = (Ebxa - Ebxb)/(Eaxa-Eaxb)
		if m > 0 : 
			#stop()
			sys.stderr.write("### WARNING : Slope > 0. Cvxhull failed")
			return frontier,frontier_energy
		thetaa = -m/(1-m)
		thetab = 1/(1-m)
		tmrf = TMRF(cmrf,[thetaa,thetab],[feat1,feat2])
		Xab = tmrf.decode()[1]
		if Xab != Xa and Xab != Xb : 
			frontier.append(Xab)
			Eaxab = cmrf.score(Xab,feat1)
			Ebxab = cmrf.score(Xab,feat2)
			frontier_energy.append((Eaxab,Ebxab))
			Q.extend([(Xa,Xab),(Xab,Xb)])
	# Calculate energy of frontier elements	
	return frontier,frontier_energy


def set_params_hmm_exp1(hmm) : 
	""" Sets the params of a hmm for sim experiment 1"""
	hmm.length = 12
	hmm.dims = [(2,3)]*hmm.length # (latent,emit) dimspace
	hmm.emit = [
		[[0.6,0.2,0.2],[0.2,0.6,0.2]]
	]*hmm.length
	hmm.trans = [
		[[0.7,0.3],[0.3,0.7]]
	]*hmm.length
	hmm.seqmap = [{'a':0,'b':1}]*hmm.length
	hmm.seqmap2 = [{0:'a',1:'b'}]*hmm.length
	hmm.featmap = [{'H':0,'B':1,'L':2}]*hmm.length
	hmm.initprob = [0.5,0.5]
	hmm.trained = True

if __name__ == '__main__' :
	hmm = HMM()
	# Set the params of the h,,
	set_params_hmm_exp1(hmm)
	cmrf = CMRF(hmm)
	seq1 = 'a'*12
	feat1 = 'HHHHLLLLHHHH'
	seq2 = 'b'*12
	feat2 = 'BBBBLLLLBBBB'

	# Find the pareto frontier
	frontier,frontier_energy = pareto_frontier(cmrf,[feat1,feat2])
	from pprint import pprint	
	pprint("Frontier is ")
	pprint(frontier)
	pprint("Frontier Energies are")
	pprint(frontier_energy)





