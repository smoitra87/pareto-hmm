""" 
A bunch of utility scripts
"""

import urllib2

def fetch_pdb(id):
  url = 'http://www.rcsb.org/pdb/files/%s.pdb' % id
  return urllib2.urlopen(url).read()



