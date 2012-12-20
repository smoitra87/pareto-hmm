""" 
HMM, CMRF and TMRF classes are defined here

@author: smoitra@cs.cmu.edu
@license: BSD

"""

import os,sys
import pylab as pl
import numpy as np


class HMM(object) : 
	""" HMM """
	def __init__(self) : 
		""" Initialized using the train method """
		self.trained = False

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





