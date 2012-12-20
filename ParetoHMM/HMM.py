""" 
HMM, CMRF and TMRF classes are defined here

@author: smoitra@cs.cmu.edu
@license: BSD

"""

import os,sys
import pylab as pl
import numpy as np
import random

class HMM(object) : 
	""" HMM """
	def __init__(self) : 
		""" Initialized using the train method """
		self.trained = False
		self.length = None
		self.emit = None # Emission probs
		self.trans = None # Transition probs
		self.dims = None # (latent,emit) dimspace
		self.featmap = None # Maps features to ids
		self.seqmap = None # Maps aa types to ids
		self.initprob = None


	def train(self,traindata) : 
		""" Train the HMM and set the params of the model"""
		pass

	def score(self,seq,obs) : 
		""" Score a given sequence and observation"""
		# Convert seq to ids
		sids = [seqmap[aa] for seqmap,aa in zip(self.seqmap,seq)]	
		# convert feature to ids
		fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]

		ll = 0.0
		# sum up scores of emission variables
		for i in xrange(self.length) : 
			ll += np.log(self.emit[i][sids[i]][fids[i]])
		
		# sum up score of latent variables
		ll += np.log(self.initprob[sids[0]])
		for i in xrange(i,self.length-1) : 
			ll+= np.log(self.trans[i][sids[i]][sids[i+1]])

		return ll


	def decode(self,obs)  : 
		""" Perform max decoding to get the optimal sequence """
		pass

	def sample(self) :
		""" Sample from the HMM"""
		if not self.trained  : 
			sys.stderr.write("HMM not yet trained. Cannot Sample!")
			return
		pass


class CMRF(HMM) : 
	""" CMRF """
	def __init__(self,hmm) : 
		""" Converts hmm to CMRF """
		pass

	def train(self,traindata) : 
		""" Train the HMM and set the params of the model"""
		pass

	def score(self,seq,obs) : 
		""" Score a given sequence and observation"""
		pass

	def decode(self,obs)  : 
		""" Perform max decoding to get the optimal sequence """
		pass

	def sample(self) :
		""" Sample from the HMM"""
		if not self.trained : 
			sys.stderr.write("HMM not yet trained. Cannot Sample!")
			return
		pass

class DataSet(object) : 
	""" A placeholder for defining training data """
	def __init__(self,dataset=1) : 
		datasetmap = {
			1 : self.create_dataset1
		}

		self.train = None
		self.test = None

		# Call dataset creator method
		datasetmap[dataset]()
	
	def create_dataset1(self) : 
		""" Create a dataset with 12 AA and beta sheets """
		pass

		
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
	hmm.featmap = [{'H':0,'B':1,'L':2}]*hmm.length
	hmm.initprob = [0.5,0.5]
	hmm.trained = True
	

if __name__ == '__main__' : 

	print("*"*10+"Running HMM"+"*"*10)	
	dataset = DataSet(dataset=1) 
	traindata = dataset.train
	hmm = HMM()
	hmm.train(traindata)
	# Set the params of the h,,
	set_params_hmm_exp1(hmm)
	cmrf = CMRF(hmm)
	seq1 = 'a'*12
	feat1 = 'HHHHLLLLHHHH'
	score1 = hmm.score(seq1,feat1)
	print('Score for seq:{0} with feat:{1} is {2}'.format(seq1,feat1,score1))

	# Now for random sequences
	for i in range(20) : 
		seq = "".join(map(lambda(x) : random.choice('ab'),range(12)));
		feat = feat1
		score = hmm.score(seq,feat)
		print('Score for seq:{0} with feat:{1} is {2}'.\
			format(seq,feat,score))




