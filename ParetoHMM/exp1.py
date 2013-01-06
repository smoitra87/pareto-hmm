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

# Constants
pdb_ss_fpath = 'data/ss_dis.txt'
pdb_seq_fpath = 'data/pdb_seqres.txt'
pdb_scop_1htm_fpath = 'data/1htm_scop.list'
pdb_scop_1aaw_fpath = 'data/1aaw_dom_scop.list'

SEEK_BUFF = 100


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
	# Count the freqs
	for c in scop_classes : 
		cseq = data[c]['seq']
		css = data[c]['ss']
		counts_seq = {}
		counts_ss = {}
		for seq in cseq : 
			for aa in seq : 
				counts_seq[aa] = counts_seq.get(aa,0)+1
		for seq in css : 
			for aa in seq : 
				counts_ss[aa] = counts_ss.get(aa,0)+1
		counts[c] = {}
		counts[c]['seq'] = counts_seq
		counts[c]['ss'] = counts_ss

	return counts
	
if __name__ == '__main__' : 
	# Create features
	p1 = Protein('1HTM',[(13,44)])	
	p2 = Protein('1AAY',[(3,33)])

	# Read 1htm scop file
	scop_1htm = parse_scop_csv('data/1htm_scop.csv')
	scop_1aay = parse_scop_csv('data/1aay_scop.csv')	

 	# Remove duplicate sequence and structures and store that \
	#as the alignment
	nondup_1htm = set(zip(scop_1htm['seq'],scop_1htm['sec']))
	nondup_1aay = set(zip(scop_1htm['seq'],scop_1htm['sec']))

	# Get sequences and ss for each of the
	data = {}
	data['1htm'] = {}
	data['1aay'] = {}
	data['1htm']['seq'] = map(itemgetter(0),nondup_1htm)
	data['1aay']['seq'] = map(itemgetter(0),nondup_1aay)
	data['1htm']['ss'] = map(itemgetter(1),nondup_1htm)
	data['1aay']['ss'] = map(itemgetter(1),nondup_1aay)

	# Learn the weights of the hmm
	base_data = calc_base(data)
	

