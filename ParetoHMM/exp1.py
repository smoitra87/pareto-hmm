""" 
In this experiment we will be enumerating the pareto frontier for 
the Zif268 multistate design problem

"""

import os,sys
from pdb import set_trace as stop
import numpy as np
import pylab as pl
import scipy as sci
import subprocess
import re
from operator import itemgetter
from itertools import product
from HMM import HMM,CMRF, TMRF, reverse_dict
import random,time, pickle
from cvxhull import pareto_frontier

# Constants
pdb_ss_fpath = 'data/ss_dis.txt'
pdb_seq_fpath = 'data/pdb_seqres.txt'
pdb_scop_1htm_fpath = 'data/1htm_scop.list'
pdb_scop_1aaw_fpath = 'data/1aaw_dom_scop.list'

SEEK_BUFF = 100
DSSP_SYM = ('H','B','E','G','I','T','S','L')
SEQ_SYM='ACDEFGHIKLMNPQRSTVWY'
BIAS=0.05
IGNORE_AA = 'XU'
SAMPLE_N = 10000
STUDY=1

def build_tied_hmm(self,align_fpath,feats) :
	""" Build a HMM using data from an alignment """

def extract_feats(pdbid) : 
	""" Extract seq and ss from the filename"""
	global pdb_ss_fpath
	pdbid = pdbid.upper()
	out = subprocess.check_output(['grep','-n',pdbid,pdb_ss_fpath])
	skip = out.split()[0].split(':')[0]
	p1 = subprocess.Popen(['head','-n','+'+str(int(skip)-1+SEEK_BUFF),\
		pdb_ss_fpath],stdout=subprocess.PIPE)
	p2 = subprocess.Popen(['tail','-n',str(SEEK_BUFF)],stdin=p1.stdout,\
		stdout=subprocess.PIPE)
	p1.stdout.close()
	output = p2.communicate()[0]
	
	# Extract the sequence and the ss
	state=0
	seq,ss,dis = '','',''
	for line in output.split('\n'):
		if state==0  and line[0] == '>' and 'sequence' in line : 
			state = 1
			continue
		if state==1 :
			if line[0] == '>' and 'secstr' in line :
				state = 2
				continue 
			else : 
				seq += line.strip('\n')
				continue	
		if state==2 :
			if line[0] == '>' and 'disorder' in line :
				state = 3
				continue 
			else : 
				ss += line.strip('\n')
				continue	
		if state == 3 : 
			if line[0] == '>' : 
				break
			else : 
				dis += line.strip('\n')
	ss = ss.replace(' ','L')
	return seq,ss,dis	

class Protein(object) :
	""" Describes a protein class """
	def __init__(self,pdbid,design_range) : 
		self.pdbid = pdbid
		self.seq,self.ss,self.dis = extract_feats(pdbid)
		if design_range == 'all' : 
			self.design_range = [(1,len(self.seq))]
		else : 
			self.design_range = design_range	
		self.des_feat = self.ss[self.design_range[0][0]-1:\
			self.design_range[0][1]]
		self.des_seq = self.seq[self.design_range[0][0]-1:\
			self.design_range[0][1]]

def read_pdb_scop_list(fpath) : 
	""" Read a pdb list file and return a list of pdb ids """
	with open(fpath) as fin : 
		pdb_list = [line.strip() for line in fin.readlines()]
	return pdb_list

def parse_scop_csv(fpath) :
	""" Parse the scop file and return the data """
	parsed = {'pdbid':[],'chain':[],'pfam':[],'seq':[],'sec':[]}
	regex = re.compile(r'\"(.*?)\"')

	with open(fpath) as fin : 
		headers = fin.readline() # skip the first line
		for line in fin.readlines() :
			if line == '\n' :
				continue
			line_list = regex.findall(line.strip('\n'))
			pdbid,chain,pfam,seqsec = line_list
			seq,sec=seqsec.split('#')
			sec = sec.replace(' ','L')
			pfam_list = [p.strip() for p in pfam.split(',')]
			for key in parsed.keys() : 
				eval('parsed["%s"].append(%s)'%(key,key))	
	return parsed


def calc_base(data)  :
	""" Calculate all the base frequencies of all chars in dataset"""
	# Scop classes
	scop_classes = data.keys()

	counts = {}
	counts['all'] = {}
	counts['all']['seq'] = {}
	counts['all']['ss'] = {}
	# Count the freqs
	for c in scop_classes : 
		cseq = data[c]['seq']
		css = data[c]['ss']
		counts_seq = {}
		counts_ss = {}
		for seq,ss in zip(cseq,css) : 
			for aa,ss_aa in zip(seq,ss) : 
				if aa in IGNORE_AA : 
					continue
				counts_seq[aa] = counts_seq.get(aa,0)+1
				counts_ss[ss_aa] = counts_ss.get(aa,0)+1
		counts[c] = {}
		counts[c]['seq'] = counts_seq
		counts[c]['ss'] = counts_ss
		for k in counts[c]['seq'].keys() : 
			counts['all']['seq'][k] =  counts['all']['seq'].get(k,0) +\
				counts[c]['seq'][k]	

		for k in counts[c]['ss'].keys() :
			counts['all']['ss'][k] =  counts['all']['ss'].get(k,0) +\
				counts[c]['ss'][k]	

	return counts
	
def learn_weights_combined(data,base_data) :
	""" Learn the emission weights by combining all data """

	weights = {}

	classes = data.keys()
	emission = {}
	trans = {}

	for aa in SEQ_SYM : 
		emission[aa] = {}
		for ss_aa in DSSP_SYM : 
			emission[aa][ss_aa] = 0
	for aa in SEQ_SYM: 
		trans[aa] = {}
		for aa2 in SEQ_SYM : 
			trans[aa][aa2] = 0
	count=0	

	# Learn the emission weights
	for c in classes : 
		cseq =  data[c]['seq'] 
		css  = data[c]['ss']
		for seq,ss in zip(cseq,css) :
			for aa,ss_aa in zip(seq,ss) : 
				if aa in IGNORE_AA : 
					continue
				try : 
					emission[aa][ss_aa] = emission[aa].get(ss_aa,0)+1
				except KeyError : 
					emission[aa] = {}
					emission[aa][ss_aa] = emission[aa].get(ss_aa,0)+1
	for key in emission.keys() : 
		count = sum(emission[key].values())
		count_ss = sum(base_data['all']['ss'].values())
		for key2 in emission[key].keys() : 
			emission[key][key2] = \
				(emission[key][key2]+BIAS*\
				base_data['all']['ss'][key2])/(count+BIAS*count_ss)
			# Add a 5% bias
	weights['emit'] = emission

	# Learn the transition weights
	for c in classes : 
		cseq =  data[c]['seq'] 
		for seq in cseq :
			for i in range(len(seq[:-1])) : 
				aa1,aa2 = seq[i:i+2]
				if aa1 in IGNORE_AA or aa2 in IGNORE_AA : 
					continue
				trans[aa1][aa2] += 1 
	
	count2 = sum(base_data['all']['seq'].values())
	for key in trans.keys() : 
		count = sum(trans[key].values())
		for key2 in trans[key].keys() : 
			trans[key][key2] = \
				(trans[key][key2]+BIAS*\
				base_data['all']['seq'][key2])/(count+BIAS*count2)
	
	initprob = {}
	for k in base_data['all']['seq'].keys()  : 
		initprob[k] =  (base_data['all']['seq'][k]+0.0)/count2

	weights['emit'] = emission
	weights['trans'] = trans
	weights['initprob'] = initprob

	return weights		

class BioExp(object) :
	""" Helps run a number of different simulation experiments """
	def __init__(self,name,tied=True,plot_all=False,**kwdargs)	:
		""" Set up the simulation study according to the parameters 
		specified"""
		self.name = name #name of experiment
		self.tied = tied
		self.plot_all = plot_all
		self.kwdargs = kwdargs
		self.ntimes=1
		# Set the random seed
	
	def execute(self)  :
		""" Execute the stored execution function """	
		# Set the random seed so that all experiments are pseudo random
		random.seed(42)
		np.random.seed(42)
		namemap = {
		'ziftied' : self.ziftied
		}	
		namemap[self.name]() # Execute ziftied

		#  Write out results
		if self.name == 'ziftied' : 
			fname = 'data/bio'+str(STUDY)+'_'+self.name+'_'+\
				self.name+'.pkl'
		else :
			fname = 'data/bio'+str(STUDY)+'_'+self.name+'.pkl'

		with open(fname,'w') as pklout : 
			pickle.dump(self,pklout)

	def ziftied(self) :
		""" Set up the toy simulation """	
		self.tasklist = []
		feats = self.kwdargs['feats']
		weights = self.kwdargs['weights']
		hmm = HMM()
		self._set_params_ziftied(hmm)
		#1/0
		cmrf = CMRF(hmm)
		for taskid in range(self.ntimes) :	
			task = Task('bio'+str(STUDY)+'_'+self.name+'_'+str(taskid),cmrf,\
				feats)				
			# Run Brute force to enumerate the frontier
#			with benchmark(task.name+'brute') as t:
#				seq,energies = self.bruteforce(cmrf,feats)			
#			task.all_seq = seq
#			task.all_seq_energy = energies
#			task.brute_time = t.elapsed			

			# Sample the frontier
			with benchmark(task.name+'sample') as t:
				seq,energies = self.sample(cmrf,feats)			
			task.sample_seq = seq
			task.sample_seq_energy = energies
			task.sample_time = t.elapsed			

			# Now run the toy simulation`
			with benchmark(task.name+'pareto') as t : 
				task.frontier,task.frontier_energy = \
					pareto_frontier(cmrf,feats)		
			if self.plot_all :
				task.plot_frontier(frontier_only = True,plot_samples=True)
			task.pareto_time = t.elapsed
			self.tasklist.append(task)	
	
	def _set_params_ziftied(self,hmm) :
		""" Sets the params of a hmm for ziftied"""
		feats = self.kwdargs['feats']
		weights = self.kwdargs['weights']
		
		hmm.length = len(feats[0])
		hmm.dims = [(len(SEQ_SYM),len(DSSP_SYM))]*hmm.length # (latent,emit) dimspace
		hmm.seqmap2 = [dict(enumerate(SEQ_SYM))]*hmm.length
		hmm.seqmap = map(reverse_dict,hmm.seqmap2)
		hmm.featmap2 = [dict(enumerate(DSSP_SYM))]*hmm.length
		hmm.featmap = map(reverse_dict,hmm.featmap2)

		hmm.emit,hmm.trans,hmm.initprob = [],[],[]
		for aa in SEQ_SYM :	
			hmm.emit.append([weights['emit'][aa][ss_aa] \
				for ss_aa in DSSP_SYM])
			hmm.trans.append([weights['trans'][aa][aa2] \
				for aa2 in SEQ_SYM])
			hmm.initprob.append(weights['initprob'][aa])

		hmm.emit = [hmm.emit]*hmm.length
		hmm.trans = [hmm.trans]*hmm.length
		hmm.trained = True

	def gen_random_dist(self,size) : 
		x = np.random.uniform(size=size)
		x = x/sum(x)
		return x

	def bruteforce(self,cmrf,feats)  :
		""" Run Brute force enumeration of the sequence space """
		feat1,feat2 = feats
		ll_list1,ll_list2 = [],[]
		seqdim = cmrf.dims[0][0]	
		seqspace ='ACDEFGHIKLMNPQRSTVWY'[:seqdim].lower()
		seq_list = ["".join(s) for s in product(seqspace,repeat=12)]
		for seq in seq_list:	
			ll_list1.append(cmrf.score(seq,feat1))
			ll_list2.append(cmrf.score(seq,feat2))

		return seq_list,zip(ll_list1,ll_list2)

	def sample(self,cmrf,feats)  :
		""" Randomly sample a section of the sequence space """
		feat1,feat2 = feats
		ll_list1,ll_list2 = [],[]
		seqdim = cmrf.dims[0][0]	
		seqlen = cmrf.length
		seqspace ='ACDEFGHIKLMNPQRSTVWY'[:seqdim]
		for i in range(SAMPLE_N):
			seq = "".join([random.choice(seqspace) for i in range(seqlen)])
			ll_list1.append(cmrf.score(seq,feat1))
			ll_list2.append(cmrf.score(seq,feat2))

		return [],zip(ll_list1,ll_list2)


		

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
		self.sample_seq = None
		self.sample_seq_energy = None
	
	def plot_frontier(self,frontier_only=False,plot_samples=True) :
		""" Plot the frontier"""
		frontier = self.frontier
		frontier_energy = self.frontier_energy
		feat1,feat2 = self.feats	
	
		pl.figure()
		if not frontier_only :	
			ll_list1,ll_list2 = zip(*self.all_seq_energy)
			pl.plot(ll_list1,ll_list2,'b*')
		if plot_samples :
			ll_list1,ll_list2 = zip(*self.sample_seq_energy)
			pl.plot(ll_list1,ll_list2,'g*')
					
		pl.plot(*zip(*sorted(frontier_energy)),color='magenta',\
			marker='*',	linestyle='dashed')
		ctr = dict(zip(set(frontier_energy),[0]*
			len(set(frontier_energy))))
		for i,e in enumerate(frontier_energy) : 
			ctr[e] += 1
			pl.text(e[0],e[1]+0.1*ctr[e],str(i),fontsize=10)
			pl.text(e[0]+0.4,e[1]+0.1*ctr[e],frontier[i],fontsize=9)	
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
		#pl.show()

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
	# Create features
	p1 = Protein('1HTM',[(13,43)])	
	p2 = Protein('1AAY',[(3,33)])

	# Read 1htm scop file
	scop_1htm = parse_scop_csv('data/1htm_scop.csv')
	scop_1aay = parse_scop_csv('data/1aay_scop.csv')	

 	# Remove duplicate sequence and structures and store that \
	#as the alignment
	nondup_1htm = set(zip(scop_1htm['seq'],scop_1htm['sec']))
	nondup_1aay = set(zip(scop_1aay['seq'],scop_1aay['sec']))

	# Get sequences and ss for each of the
	data = {}
	data['1htm'] = {}
	data['1aay'] = {}
	data['1htm']['seq'] = map(itemgetter(0),nondup_1htm)
	data['1aay']['seq'] = map(itemgetter(0),nondup_1aay)
	data['1htm']['ss'] = map(itemgetter(1),nondup_1htm)
	data['1aay']['ss'] = map(itemgetter(1),nondup_1aay)

	########## Learn the weights of the hmm #############3

	# Calculate the base frequencies of the data
	base_data = calc_base(data)
	
	# Learn emission weights
	weights = learn_weights_combined(data,base_data)	

	# Assert that the weights are correct
	for k in weights.keys() : 
		wt = weights[k]
		if k == 'initprob' : continue
		for k2 in wt.keys() : 
			assert round(sum(wt[k2].values())-1.0,7) == 0.0

	# Learn an HMM and run the experiments
	sim = BioExp('ziftied',feats=[p1.des_feat,p2.des_feat],weights=weights,plot_all=True)	
	sim.execute()	
#
