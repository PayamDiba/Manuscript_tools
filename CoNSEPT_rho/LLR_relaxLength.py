
"""
This object scans the sequence with provided TFs and calculate LLR on each position

build the object and just run "scan" method.

TODO: make sure rev complement is done correctly here
"""
from collections import defaultdict
import numpy as np
import pandas as pd

class calculate_LLR(object):
    def __init__(self, pathSeqFile, tfDict):
        self.seqNames, self.seqs = self.read_seq_file(pathSeqFile) # python List #s * 2 * L * 4
        self.tfDict = tfDict # #TFs * 10 * 4
        self.nSeqs = len(self.seqs)
    def read_seq_file (self, seq_file):
        """
        This method reads all the sequences from the sequence file
        """
        df = pd.read_csv(seq_file, header=None, index_col=None, sep='\n')
        df = df.values
        allSeq = []
        seqLabel = []
        self.maxLength = 0

        for l in df:
            currLine = l[0]

            if currLine[0] != '>':
                rev_currLine = self.reverse_complement(currLine)
                allSeq.append([self.encode_seq(currLine),self.encode_seq(rev_currLine)])
                self.maxLength = max(self.maxLength, len(currLine))

            else:
                seqLabel.append(currLine[1:])

        return seqLabel, allSeq

    def reverse_complement(self,seq):
        ret = ''
        for b in seq:
            if b.capitalize() == 'A':
                ret += 'T'
            elif b.capitalize() == 'T':
                ret += 'A'
            elif b.capitalize() == 'C':
                ret += 'G'
            elif b.capitalize() == 'G':
                ret += 'C'
            elif b.capitalize() == 'N':
                ret += 'N'

        return ret

    def encode_seq(self, seq_string):
        """
        Input: one DNA sequence as string of length L
        returns: np.array(L,4), one-hot encoded sequence

        4 columns: A, C, G, T
        """

        ret = np.zeros((len(seq_string),4))
        for ib, b in enumerate(seq_string):
            if b.capitalize() == 'A':
                ret[ib, 0] = 1
            elif b.capitalize() == 'C':
                ret[ib, 1] = 1
            elif b.capitalize() == 'G':
                ret[ib, 2] = 1
            elif b.capitalize() == 'T':
                ret[ib, 3] = 1
            elif b.capitalize() == 'N':
                pass

            # else b == 'N': let it remain zero

        return ret

    def scan(self, training):
        """
        Note that both strands are in their correct order, so no additional flip is required
        """

        ret = defaultdict(dict)
        #np.zeros((self.nSeqs,2,self.len_-9))

        for sname, s in zip(self.seqNames, self.seqs):
            currLen = np.shape(s)[1]
            offset = self.maxLength - currLen

            if not training:
                all_offSets = [0]
            else:
                if offset+1 > 10:
                    all_offSets = np.random.choice(range(offset+1),10, replace = False).tolist()
                else:
                    all_offSets = list(range(offset+1))
                if 0 in all_offSets:
                    all_offSets.remove(0)
                all_offSets.insert(0,0) #make sure to have offset zero, and make sure that it's the first offset
            for currOS in all_offSets:
                ret[sname][currOS] = {}
                for tfName in self.tfDict:
                    tf = self.tfDict[tfName]
                    ret[sname][currOS][tfName] = np.zeros((self.maxLength-9,2))
                    for strandID, currS in enumerate(s):
                        bMin = 0
                        bMax = 10
                        while(bMax <= currLen):
                            segment = currS[bMin:bMax,:]
                            if strandID == 0:
                                pass
                            elif strandID == 1:
                                tf = np.flip(tf,axis=0)
                            else:
                                print("Error!!!")

                            score = np.multiply(segment, tf)
                            score = self.maxBind(score)
                            maxScore = self.maxBind(tf)
                            score = np.true_divide(score,maxScore)


                            ret[sname][currOS][tfName][bMin + currOS, strandID] = score
                            bMin+=1
                            bMax+=1
        return ret

    def maxBind(self,tf):
        ret = 1
        for b in tf:
            ret = ret * max(b)

        return ret
