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
import time
import pickle
DEBUG=0

STUDY='3'


class SimExp(object) :
	""" Helps run a number of different simulation experiments """
	def __init__(self,name,ntimes=10,length=12,seqstates=2,tied=True\
		,plot_all=False)	:
		""" Set up the simulation study according to the parameters 
		specified"""
		self.name = name #name of experiment
		self.ntimes = ntimes # number of times to repeat the exp
		self.length = length
		self.seqstates = seqstates
		self.tied = tied
		self.plot_all = plot_all

		# Set the random seed
	
	def execute(self)  :
		""" Execute the stored execution function """	
		# Set the random seed so that all experiments are pseudo random
		random.seed(42)
		namemap = {
		'toy' : self.toy,
		'randprobs' : self.randprobs,
		'randprobstied' : self.randprobstied
		}	
		namemap[self.name]()
		with open('data/sim'+str(STUDY)+'_'+self.name+'.pkl','w') as \
			pklout : 
			pickle.dump(self,pklout)

	def toy(self) :
		""" Set up the toy simulation """	
		self.tasklist = []
		feats = self.get_feats_standard()
		hmm = HMM()
		self._set_params_toy(hmm)
		cmrf = CMRF(hmm)
		for taskid in range(self.ntimes) :	
			task = Task('sim'+STUDY+'_'+self.name+'_'+str(taskid),cmrf,\
				feats)				
			# Run Brute force to enumerate the frontier
			with benchmark(task.name+'brute') as t:
				seq,energies = self.bruteforce(cmrf,feats)			
			task.all_seq = seq
			task.all_seq_energy = energies
			task.brute_time = t.elapsed			

			# Now run the toy simulation`
			with benchmark(task.name+'pareto') as t : 
				task.frontier,task.frontier_energy = \
					pareto_frontier(cmrf,feats)		
			if self.plot_all :
				task.plot_frontier()
			task.pareto_time = t.elapsed
			self.tasklist.append(task)	
	
	def _set_params_toy(self,hmm) :
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


	def randprobs(self) : 
		""" Set up a version of toy with non-trivial surface """
		self.tasklist = []
		feats = self.get_feats_standard()
		hmm = HMM()
		self._set_params_randprobs(hmm)
		cmrf = CMRF(hmm)
		for taskid in range(self.ntimes) :	
			task = Task('sim'+STUDY+'_'+self.name+'_'+str(taskid),cmrf,\
				feats)				
			# Run Brute force to enumerate the frontier
			with benchmark(task.name+'brute') as t:
				seq,energies = self.bruteforce(cmrf,feats)			
			task.all_seq = seq
			task.all_seq_energy = energies
			task.brute_time = t.elapsed			

			# Now run the toy simulation`
			with benchmark(task.name+'pareto') as t : 
				task.frontier,task.frontier_energy = \
					pareto_frontier(cmrf,feats)		
			if self.plot_all :
				task.plot_frontier()
			task.pareto_time = t.elapsed
			self.tasklist.append(task)	

	def _set_params_randprobs(self,hmm) :
		""" Sets the params of a hmm for sim experiment 1"""
		hmm.length = 12
		hmm.dims = [(2,3)]*hmm.length # (latent,emit) dimspace
		hmm.emit = [
			[[0.334,0.272,0.394],[0.477,0.093,0.430]]
		]*hmm.length
		hmm.trans = [
			[[0.483,0.517],[0.589,0.411]]
		]*hmm.length

		hmm.seqmap = [{'a':0,'b':1}]*hmm.length
		hmm.seqmap2 = [{0:'a',1:'b'}]*hmm.length
		hmm.featmap = [{'H':0,'B':1,'L':2}]*hmm.length
		hmm.initprob = [0.5,0.5]
		hmm.trained = True


	def set_params_randprobstied(hm) : 
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


	def randprobstied(self) : 
		""" Run many iterations of toy with random probs  """
		pass

	def get_feats_standard(self) :
		""" Set the features to be standard """ 
		feats = [
			 'HHHHLLLLHHHH',
			 'BBBBLLLLBBBB'
		]
		return feats

	def gen_random_dist(size) : 
		x = np.random.uniform(size=size)
		x = x/sum(x)
		return x

	def bruteforce(self,cmrf,feats)  :
		""" Run Brute force enumeration of the sequence space """
		feat1,feat2 = feats
		ll_list1,ll_list2 = [],[]
		
		seq_list = ["".join(s) for s in product('ab',repeat=12)]
		for seq in seq_list:	
			ll_list1.append(cmrf.score(seq,feat1))
			ll_list2.append(cmrf.score(seq,feat2))

		return seq_list,zip(ll_list1,ll_list2)
	


	def mcmc(self) : 
		""" Run mcmc on the same problem instance """
		### TODO
		pass



class Task(object) :
	""" Stores the result object and other tracking features"""
	def __init__(self,name,hmm,feats) :
		self.hmm = hmm
		self.name = name
		self.frontier = None
		self.frontier_energy = None
		self.all_seq = None
		self.all_seq_energy = None
		self.feats = feats
	
	def plot_frontier(self,frontier_only=False) :
		""" Plot the frontier"""
		ll_list1,ll_list2 = zip(*self.all_seq_energy)
		frontier = self.frontier
		frontier_energy = self.frontier_energy
		feat1,feat2 = self.feats	
	
		pl.figure()
		if not frontier_only :	
			pl.plot(ll_list1,ll_list2,'b*')
		pl.plot(*zip(*sorted(frontier_energy)),color='magenta',\
			marker='*',	linestyle='dashed')
		ctr = dict(zip(set(frontier_energy),[0]*
			len(set(frontier_energy))))
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
		pic_dir = '../docs/tex/pics/'
		pl.savefig(pic_dir+self.name+'.pdf')
		pl.savefig(pic_dir+self.name+'.png')

class benchmark(object):
	def __init__(self,name):
		self.name = name
	def __enter__(self):
		self.start = time.time()
		return self
	def __exit__(self,ty,val,tb):
		end = time.time()
		self.elapsed =  end-self.start
		print("%s : %0.3f seconds" % (self.name, end-self.start))

if __name__ == '__main__' : 

	# Find the pareto frontier
	#frontier,frontier_energy = pareto_frontier(cmrf,[feat1,feat2])

	# Run sim experiment 1 - toy
	#sim1 = SimExp('toy',plot_all=True)	
	sim1 = SimExp('toy',ntimes=1,plot_all=True)	
	sim1.execute()	

	# Run sim experiment 2 - 	
	sim2 = SimExp('randprobs',ntimes=1,plot_all=True)
	sim2.execute()
