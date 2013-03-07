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
from cvxhull import pareto_frontier 
import random
class BoostedHMM(object) :
	""" Generates boosted HMMs """

	def __init__(self)  :
		self.k = 4
		self.length = 12
		self.kseqlist = []
		for i in range(self.k) : 
			seq = "".join(map(lambda x:random.choice('ab'),range(self.length)))
			self.kseqlist.append(seq)

		self.smoothfac = 0.001

		self.hmm1 = self.learn_hmm(self.kseqlist)	
		self.hmm2 = self.learn_hmm(self.kseqlist[::-1])

	def learn_hmm(self,seqlist) : 
		""" Learns hmm from seqlist"""
		hmm = HMM()
		hmm.length = self.length
		hmm.dims = [(2,1)]*hmm.length # (latent,emit) dimspace
		hmm.emit = [
			[[1.0],[1.0]]
		]*hmm.length
			
		hmm.seqmap = [{'a':0,'b':1}]*hmm.length
		hmm.seqmap2 = [{0:'a',1:'b'}]*hmm.length
		hmm.featmap = [{'H':0}]*hmm.length
		hmm.initprob = [0.5,0.5]
		hmm.trained = True
		hmm.alphabet = 'ab'	

		# Calculate HMM transition probabilities
		hmm.trans = [
			[[0.7,0.3],[0.3,0.7]]
		]*hmm.length
		
		counts,counts2 = [],[]
		for i in range(len(seqlist[0])) :
			counts.append({})
			counts2.append({})

		for i,seq in enumerate(seqlist) : 
			for j,aa in enumerate(seq) : 
				counts[j][aa] = counts[j].get(aa,0) + self.k - i

		for i,seq in enumerate(seqlist) : 
			for j,aa in enumerate(seq[:-1]) : 
				counts2[j][seq[j:j+2]] = counts2[j].get(seq[j:j+2],0) + self.k - i
		
		hmm.trans = []

		for i in range(len(seqlist[0])-1) :
			hmm.trans.append([])
			for j,aa1 in enumerate(hmm.alphabet) : 
				hmm.trans[-1].append([])
				for k,aa2 in enumerate(hmm.alphabet) :
					val = (counts2[i].get(aa1+aa2,0)+self.smoothfac) / (counts[i].get(aa1,0)+self.smoothfac*len(hmm.alphabet))
					hmm.trans[-1][-1].append(val)
		return hmm

if __name__ == '__main__' : 
	b = BoostedHMM()
	hmm1,hmm2 = b.hmm1,b.hmm2
	# Set the params of the hmm
	cmrf1 = CMRF(hmm1)
	cmrf2 = CMRF(hmm2)
	feat = 'HHHHHHHHHHHH'

	# Plot the entire sequence space
	ll_list1,ll_list2 = [],[]
	for seq in product('ab',repeat=12):	
		ll_list1.append(cmrf1.score(seq,feat))
		ll_list2.append(cmrf2.score(seq,feat))
	ll_list3,ll_list4 = [],[]
	for seq in b.kseqlist:	
		ll_list3.append(cmrf1.score(seq,feat))
		ll_list4.append(cmrf2.score(seq,feat))


	pl.figure()
	pl.plot(ll_list1,ll_list2,'b*')
	pl.plot(ll_list3,ll_list4,'r*')

	pl.xlabel('Energy1:')
	pl.ylabel('Energy2:')
	pl.title('Energy Plot')
	xmin,xmax = pl.xlim()
	ymin,ymax = pl.ylim()
	pl.xlim(-2,xmax)
	pl.ylim(-2,ymax)
	pl.axvline()
	pl.axhline()
	pl.savefig('../docs/tex/pics/sim4_nosmooth_3.png')
	pl.show()
