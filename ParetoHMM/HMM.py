""" 
HMM, CMRF and TMRF classes are defined here

@author: smoitra@cs.cmu.edu
@license: BSD

"""

import os,sys
import pylab as pl
import numpy as np
import random


def reverse_dict(d)	:	
	""" Reverses a dictionary making the keys be the values"""
	return dict([(v,k) for k,v in d.viewitems()])	

class HMM(object) : 
	""" HMM """
	def __init__(self) : 
		""" Initialized using the train method """
		self.trained = False
		self.length = None
		self.emit = None # Emission probs (seq,feat)
		self.trans = None # Transition probs
		self.dims = None # (latent,emit) dimspace
		self.featmap = None # Maps features to ids
		self.seqmap = None # Maps aa types to ids
		self.initprob = None
		self.seqmap2 = None
		self.fratmap2 = None

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
		for i in xrange(self.length-1) : 
			ll+= np.log(self.trans[i][sids[i]][sids[i+1]])

		return ll


	def decode(self,obs)  : 
		""" Perform max decoding to get the optimal sequence """
		# convert feature to ids
		fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]
		V = []
		Ptr = []
		# For the first element
		V.append(map(lambda(k):np.log(self.emit[0][k][fids[0]])\
			+np.log(self.initprob[k]),range(self.dims[0][0])))

		# For the rest of the elements
		for i in range(1,hmm.length) :
			ptr_k = []; V_k = [];
			for k in range(self.dims[i][0]) :
				X = map(lambda(j):\
					np.log(self.trans[i-1][j][k])+V[i-1][j],\
					range(self.dims[i-1][0]))
				maxX = max(X)
				ptr_k.append(X.index(maxX))
				V_k.append(np.log(self.emit[i][k][fids[i]])+maxX)
			Ptr.append(ptr_k)
			V.append(V_k)

		# retreive the optimal sequence
		seq_max = []
		seq_max.append(V[-1].index(max(V[-1])))
		energy_max = V[-1][seq_max[-1]]
		for ptr_k in reversed(Ptr) :
			seq_max.append(ptr_k[seq_max[-1]])
		seq_max.reverse()
		
		# map it back to the original sequence
		decode_seq = "".join([self.seqmap2[i][seqid] \
			for i,seqid in enumerate(seq_max)])
	
		return energy_max,decode_seq

	def sample(self) :
		""" Sample from the HMM"""
		if not self.trained  : 
			sys.stderr.write("HMM not yet trained. Cannot Sample!")
			return
		pass

class CMRF(object) : 
	""" CMRF """
	def __init__(self,hmm) : 
		""" Converts hmm to CMRF """
		self.trained = hmm.trained
		self.length = hmm.length
		self.emitpsi = hmm.emit # Emission probs (seq,feat)
		self.seqpsi = hmm.trans # Transition probs
		self.dims = hmm.dims # (latent,emit) dimspace
		self.featmap = hmm.featmap # Maps features to ids
		self.seqmap = hmm.seqmap # Maps aa types to ids
		self.seqphi = [np.ones(dim[0]) for dim in hmm.dims]
		self.featphi =[np.ones(dim[1]) for dim in hmm.dims] 

		# maps from id to seq
		self.seqmap2 = map(reverse_dict,self.seqmap)
		self.featmap2 = map(reverse_dict,self.featmap)

	def score(self,seq,obs) : 
		""" Score a given sequence and observation"""
		# Convert seq to ids
		sids = [seqmap[aa] for seqmap,aa in zip(self.seqmap,seq)]	
		# convert feature to ids
		fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]

		ll = 0.0
		# sum up scores of emission variables
		for i in xrange(self.length) : 
			ll += np.log(self.emitpsi[i][sids[i]][fids[i]])
		
		# sum up score of latent variables
		for i in xrange(self.length-1) : 
			ll+= np.log(self.seqpsi[i][sids[i]][sids[i+1]])

		# sum up the scores of the nodes
		for i in xrange(self.length) : 
			ll+=  np.log(self.seqphi[i][sids[i]])
			ll+= np.log(self.featphi[i][fids[i]])
		return ll


	def decode(self,obs)  : 
		""" Perform max decoding to get the optimal sequence """
		# convert feature to ids
		fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]
		V = []
		Ptr = []
		# For the first element
		V.append(map(lambda(k):np.log(self.emitpsi[0][k][fids[0]])\
			+np.log(self.seqphi[0][k]),range(self.dims[0][0])))

		# For the rest of the elements
		for i in range(1,hmm.length) :
			ptr_k = []; V_k = [];
			for k in range(self.dims[i][0]) :
				X = map(lambda(j):np.log(self.seqpsi[i-1][j][k])+V[i-1][j],\
					range(self.dims[i-1][0]))
				maxX = max(X)
				ptr_k.append(X.index(maxX))
				V_k.append(np.log(self.seqphi[i][k])+\
					np.log(self.emitpsi[i][k][fids[i]])+maxX)
			Ptr.append(ptr_k)
			V.append(V_k)

		# retreive the optimal sequence
		seq_max = []
		seq_max.append(V[-1].index(max(V[-1])))
		energy_max = V[-1][seq_max[-1]]
		for ptr_k in reversed(Ptr) :
			seq_max.append(ptr_k[seq_max[-1]])
		seq_max.reverse()
		
		# map it back to the original sequence
		decode_seq = "".join([self.seqmap2[i][seqid] \
			for i,seqid in enumerate(seq_max)])
	
		return energy_max,decode_seq

	def sample(self) :
		""" Sample from the HMM"""
		if not self.trained : 
			sys.stderr.write("HMM not yet trained. Cannot Sample!")
			return
		pass

class TMRF(object) : 
	""" The Tree MRF which can be used for decoding """
	def __init__(self,cmrf,thetalist,featlist) : 
		""" Converts A list of cmrfs to a joint TMRF """
		self.nStates = len(thetalist)
		assert(self.nStates>1)
		assert(len(thetalist)==len(featlist))
		self.thetalist = thetalist
		self.featlist = featlist
		self.length = cmrf.length
		self.dims = cmrf.dims # (latent,emit) dimspace
		self.featmap = cmrf.featmap # Maps features to ids
		self.seqmap = cmrf.seqmap # Maps aa types to ids
		self.seqpsi = cmrf.seqpsi
		self.seqphi = cmrf.seqphi
		self.featphi = cmrf.featphi
		self.emitpsi = cmrf.emitpsi

		# maps from id to seq
		self.seqmap2 = map(reverse_dict,self.seqmap)
		self.featmap2 = map(reverse_dict,self.featmap)

	def score_seq(self,seq) : 
		""" Score a given sequence and observation"""
		# Convert seq to ids
		sids = [seqmap[aa] for seqmap,aa in zip(self.seqmap,seq)]	
		# convert feature to ids
		fid_list  = []
		for obs in self.featlist : 
			fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]
			fid_list.append(fids)
		ll = 0.0
		# sum up scores of emission variables
		for i in xrange(self.length) : 
			for j in xrange(self.nStates) :
				fids,theta = fid_list[j],self.thetalist[j]
				ll += theta*np.log(self.emitpsi[i][sids[i]][fids[i]])	
			
		for i in xrange(self.length-1) : 
			ll+= np.log(self.seqpsi[i][sids[i]][sids[i+1]])

		# sum up the scores of the nodes
		for i in xrange(self.length) : 
			ll+=  np.log(self.seqphi[i][sids[i]])
			ll+= np.log(self.featphi[i][fids[i]])
		return ll
		

	def decode(self)  : 
		""" Perform max decoding to get the optimal sequence """
		# convert feature to ids
		fid_list  = []
		for obs in self.featlist : 
			fids = [featmap[f] for featmap,f in zip(self.featmap,obs)]
			fid_list.append(fids)
		
		V = []
		Ptr = []
		# For the first element
		v_k = []
		for k in xrange(self.dims[0][0]) :
			X = np.log(self.seqphi[0][k]);
			for j in xrange(self.nStates) :
				fids,theta = fid_list[j],self.thetalist[j]
				X += theta*np.log(self.emitpsi[0][k][fids[0]])
			v_k.append(X)
		V.append(v_k)

		# For the rest of the elements
		for i in range(1,hmm.length) :
			ptr_k = []; V_k = [];
			for k in range(self.dims[i][0]) :
				X = map(lambda(j):np.log(self.seqpsi[i-1][j][k])\
					+V[i-1][j],range(self.dims[i-1][0]))
				maxX = max(X)
				ptr_k.append(X.index(maxX))
				Y = sum([theta*np.log(self.emitpsi[i][k][fids[i]]) \
					for theta,fids in zip(self.thetalist,fid_list)])
				V_k.append(Y+maxX)
			Ptr.append(ptr_k)
			V.append(V_k)

		# retreive the optimal sequence
		seq_max = []
		seq_max.append(V[-1].index(max(V[-1])))
		energy_max = V[-1][seq_max[-1]]
		for ptr_k in reversed(Ptr) :
			seq_max.append(ptr_k[seq_max[-1]])
		seq_max.reverse()
		
		# map it back to the original sequence
		decode_seq = "".join([self.seqmap2[i][seqid] \
			for i,seqid in enumerate(seq_max)])
	
		return energy_max,decode_seq

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
	hmm.seqmap2 = [{0:'a',1:'b'}]*hmm.length
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
	seq2 = 'b'*12
	feat2 = 'BBBBLLLLBBBB'
	seq3 = 'a'*6+'b'*6
	feat3 = 'HHHHLLLLBBBB'


	score1 = hmm.score(seq1,feat1)
	print('Score for seq:{0} with feat:{1} is {2}'.format(seq1,feat1,score1))

	# Now for random sequences
	for i in range(20) : 
		seq = "".join(map(lambda(x) : random.choice('ab'),range(12)));
		feat = feat1
		score = hmm.score(seq,feat)
		print('Score for seq:{0} with feat:{1} is {2}'.\
			format(seq,feat,score))


	print("*"*10+"Decoding HMM"+"*"*10)
	score,seq = hmm.decode(feat1)	
	print("Decoded seq:{} energy:{} feat:{}".format(seq,score,feat1))
	score,seq = hmm.decode(feat3)	
	print("Decoded seq:{} energy:{} feat:{}".format(seq,score,feat3))


	print('*'*10+"Working with CMRF"+"*"*10)
	cmrf = CMRF(hmm)	
	print("CMRF seq:{} feat:{} score:{}".\
		format(seq1,feat1,cmrf.score(seq1,feat1)))
	print("CMRF seq:{} feat:{} score:{}".\
		format(seq2,feat2,cmrf.score(seq2,feat2)))
	print("CMRF seq:{} feat:{} score:{}".\
		format(seq3,feat3,cmrf.score(seq3,feat3)))
	print("CMRF seq:{} feat:{} score:{}".\
		format(seq2,feat1,cmrf.score(seq2,feat1)))
	print("CMRF seq:{} feat:{} score:{}".\
		format(seq3,feat1,cmrf.score(seq3,feat1)))

	print("*"*10+"Decoding CMRF"+"*"*10)
	score,seq = cmrf.decode(feat1)	
	print("Decoded seq:{} energy:{} feat:{}".format(seq,score,feat1))
	score,seq = cmrf.decode(feat3)	
	print("Decoded seq:{} energy:{} feat:{}".format(seq,score,feat3))

	print('*'*10+"Working with TMRF"+"*"*10)
	tmrf = TMRF(cmrf,[0.0,1.0],[feat1,feat2])	
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq1,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq1)))
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq2,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq2)))
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq3,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq3)))
	tmrf = TMRF(cmrf,[1.0,0.0],[feat1,feat2])	
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq1,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq1)))
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq2,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq2)))
	print("TMRF seq:{} feat1:{} feat2:{} theta:{} score:{}".\
		format(seq3,feat1,feat2,tmrf.thetalist,tmrf.score_seq(seq3)))

	print("*"*10+"Decoding TMRF"+"*"*10)
	theta = [1.0,0.0]
	tmrf = TMRF(cmrf,theta,[feat1,feat2])	
	score,seq = tmrf.decode()	
	print("Decoded seq:{} energy:{} feat1:{} feat2:{} theta:{}".\
		format(seq,score,feat1,feat2,theta))
	theta = [0.0,1.0]
	tmrf = TMRF(cmrf,theta,[feat1,feat2])	
	score,seq = tmrf.decode()	
	print("Decoded seq:{} energy:{} feat1:{} feat2:{} theta:{}".\
		format(seq,score,feat1,feat2,theta))
	theta = [0.5,0.5]
	tmrf = TMRF(cmrf,theta,[feat1,feat2])	
	score,seq = tmrf.decode()	
	print("Decoded seq:{} energy:{} feat1:{} feat2:{} theta:{}".\
		format(seq,score,feat1,feat2,theta))
	theta = [0.4,0.6]
	tmrf = TMRF(cmrf,theta,[feat1,feat2])	
	score,seq = tmrf.decode()	
	print("Decoded seq:{} energy:{} feat1:{} feat2:{} theta:{}".\
		format(seq,score,feat1,feat2,theta))
	theta = [0.6,0.4]
	tmrf = TMRF(cmrf,theta,[feat1,feat2])	
	score,seq = tmrf.decode()	
	print("Decoded seq:{} energy:{} feat1:{} feat2:{} theta:{}".\
		format(seq,score,feat1,feat2,theta))


