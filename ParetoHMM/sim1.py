"""
Simulation Study 1 :
Create a 12 length protein

@author: smoitra@cs.cmu.edu
@license: BSD
"""
import os,sys
import pylab as pl
import numpy as np
import scipy as sci
from HMM import HMM,CMRF,TMRF
from itertools import product

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


	
	
	# Plot the entire sequence space
	ll_list1,ll_list2 = [],[]
	for seq in product('ab',repeat=12):	
		ll_list1.append(cmrf.score(seq,feat1))
		ll_list2.append(cmrf.score(seq,feat2))

	pl.figure()
	pl.plot(ll_list1,ll_list2,'b*')
	pl.xlabel('Energy:'+feat1)
	pl.ylabel('Energy:'+feat2)
	pl.title('Energy Plot')
	xmin,xmax = pl.xlim()
	ymin,ymax = pl.ylim()
	pl.xlim(xmin,2)
	pl.ylim(ymin,2)
	pl.axvline()
	pl.axhline()
