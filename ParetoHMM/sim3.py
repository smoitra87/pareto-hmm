"""
Simulation Study 2 :
Run a number of different simulation studies

@author: smoitra@cs.cmu.edu
@license: BSD
"""
import os,sys
import pylab as pl
import numpy as np
import scipy as sci
from HMM import HMM,CMRF,TMRF
from itertools import product
from cvxhull import pareto_frontier 
import random
DEBUG=0


class SimExp(object) :
	""" Helps run a number of different simulation experiments """
	def __init__(self,name,ntimes=10,length=12,seqstates=2,tied=True\
		)	:
		""" Set up the simulation study according to the parameters 
		specified"""
		self.name = name #name of experiment
		self.ntimes = ntimes # number of times to repeat the exp
		self.length = length
		self.seqstates = seqstates
		self.tied = tied
		self.hmm = HMM()	
		self.namemap = {
		'toy' : self.toy
		'randprobs' : self.randprobs
		'randprobstied' : self.randprobstied
		}	
		self.namemap[self.name]()
		self.execute()

	def toy(self) :
		""" Set up the toy simulation """
		pass

	def randprobs(self) : 
		""" Set up a version of toy with non-trivial surface """
		pass

	def randprobstied(self) : 
		""" Run many iterations of toy with random probs  """

	def set_feats_standard(self) : 
		self.feats = [
			feat1 = 'HHHHLLLLHHHH'
			feat2 = 'BBBBLLLLBBBB'
		]
	def gen_random_dist(size) : 
		x = np.random.uniform(size=size)
		x = x/sum(x)
		return x
	
	def set_params_hmm(hmm) : 
		""" Sets the params of a hmm for sim experiment 1"""
		hmm.length = 12
		hmm.dims = [(2,3)]*hmm.length # (latent,emit) dimspace
		hmm.emit = [
			[gen_random_dist(3),gen_random_dist(3)]
		]*hmm.length
		hmm.trans = [
			[gen_random_dist(2),gen_random_dist(2)]
		]*hmm.length
		hmm.seqmap = [{'a':0,'b':1}]*hmm.length
		hmm.seqmap2 = [{0:'a',1:'b'}]*hmm.length
		hmm.featmap = [{'H':0,'B':1,'L':2}]*hmm.length
		hmm.initprob = [0.5,0.5]
		hmm.trained = True

	def bruteforce(self)  :
		""" Run Brute force enumeration of the sequence space """
		feat1,feat2 = self.feats
		ll_list1,ll_list2 = [],[]
		
		seq_list = ["".join(s) for s in product('ab',repeat=12)]
		for seq in seq_list:	
			ll_list1.append(cmrf.score(seq,feat1))
			ll_list2.append(cmrf.score(seq,feat2))
	
		min_feat1id = ll_list1.index(min(ll_list1))
		min_feat2id = ll_list2.index(min(ll_list2))


	def mcmc(self) : 
		""" Run mcmc on the same problem instance """
		### TODO
		pass


if __name__ == '__main__' : 

	# Plot the entire sequence space
	ll_list1,ll_list2 = [],[]
	seq_list = ["".join(s) for s in product('ab',repeat=12)]
	for seq in seq_list:	
		ll_list1.append(cmrf.score(seq,feat1))
		ll_list2.append(cmrf.score(seq,feat2))

	min_feat1id = ll_list1.index(min(ll_list1))
	min_feat2id = ll_list2.index(min(ll_list2))

	# Find the pareto frontier
	frontier,frontier_energy = pareto_frontier(cmrf,[feat1,feat2])

	# Plot only the frontier
	pl.figure()
	pl.plot(*zip(*sorted(frontier_energy)),color='magenta',marker='*',\
		linestyle='dashed')
	ctr = dict(zip(set(frontier_energy),[0]*len(set(frontier_energy))))
	for i,e in enumerate(frontier_energy) : 
		ctr[e] += 1
		pl.text(e[0],e[1]+0.1*ctr[e],str(i),fontsize=10)
		pl.text(e[0]+0.1,e[1]+0.1*ctr[e],frontier[i],fontsize=9)	
	pl.xlabel('Energy:'+feat1)
	pl.ylabel('Energy:'+feat2)
	pl.title('Energy Plot')
	xmin,xmax = pl.xlim()
	ymin,ymax = pl.ylim()
	pl.xlim(xmin,xmax)
	pl.ylim(ymin,ymax)
	pl.savefig('../docs/tex/pics/sim2_frontier.pdf')

	# Plot all the points 
	pl.figure()
	pl.plot(ll_list1,ll_list2,'b*')
	pl.plot(*sorted(zip(*frontier_energy)),color='magenta',marker='*',\
		linestyle='dashed')
	pl.xlabel('Energy:'+feat1)
	pl.ylabel('Energy:'+feat2)
	pl.title('Energy Plot')
	xmin,xmax = pl.xlim()
	ymin,ymax = pl.ylim()
	pl.xlim(-2,xmax)
	pl.ylim(-2,ymax)
	pl.axvline()
	pl.axhline()
	pl.savefig('../docs/tex/pics/sim2_all.pdf')
	pl.show()
