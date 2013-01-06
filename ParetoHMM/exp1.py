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

# Constants
pdb_ss_fpath = 'data/ss_dis.txt'
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
		self.design_range = design_range	
		self.des_feat = self.ss[design_range[0][0]-1:design_range[0][1]]
		self.des_seq = self.seq[design_range[0][0]-1:design_range[0][1]]
	
if __name__ == '__main__' : 
	p1 = Protein('1HTM',[(13,44)])	
	p2 = Protein('1AAY',[(3,33)])
