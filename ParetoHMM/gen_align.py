"""
Generate the aligns 
"""

from Bio import SeqIO,AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein


map1htm = 'data/PF00509_full_1htm_map.fasta'
full1htm = 'data/PF00509_full.fasta'
full1htm_filt = 'data/full_1htm.fasta'
map1aay = 'data/1aay_full.map'
full1aay = 'data/PF00096_full.fasta'
full1aay_filt = 'data/full_1aay.fasta'

GAP_LIM=0.3 # Keep only sequences with less than 30% gap

def gen_filter(pos,fpath) : 
	for seqr in SeqIO.parse(fpath,"fasta") : 
		seq = Seq("".join([seqr.seq[p] for p in pos]),generic_protein)
		if (seq.count('-')+0.0)/len(seq) > GAP_LIM : 
			continue
		seqr2 = SeqRecord(seq,id=seqr.id)
		yield seqr2

def gen_filter2(pos,fpath) : 
	for seqr in SeqIO.parse(fpath,"fasta") : 
		seq = ""
		for p in pos :
			if p == '-' :
				seq += '-'
			else :
				seq += seqr.seq[int(p)]
		seq = Seq(seq,generic_protein)
		if (seq.count('-')+0.0)/len(seq) > GAP_LIM : 
			continue
		seqr2 = SeqRecord(seq,id=seqr.id)
		yield seqr2

if __name__ == '__main__' : 
	
	#### Create the 1htm alignment
	# Read the map
#	seqs = [seqr.seq for seqr in SeqIO.parse(map1htm,"fasta")]
#	seq = seqs[1]
#	pos = [i for i,aa in enumerate(seq) if aa != '-']
#
#	# Load the large alignment
#	gen_seqs = gen_filter(pos,full1htm)
#	SeqIO.write(gen_seqs,full1htm_filt,"fasta")
	

	#### Create the 1aay alignment
	# Read the map
	with open(map1aay) as fin : 
		pos = [line.strip().split()[0] for line in fin.readlines()]

	# Load the large alignment
	gen_seqs =  gen_filter2(pos,full1aay) 
	SeqIO.write(gen_seqs,full1aay_filt,"fasta")
