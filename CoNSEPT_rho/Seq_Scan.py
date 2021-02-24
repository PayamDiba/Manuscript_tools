import numpy as np
import pandas as pd
from LLR_relaxLength import calculate_LLR


class Seq(object):
    def __init__(self, seq_file, PWM, expression_file, TF_file, nEnhancers, nBins, nTrain, nValid, nTest, training = True):
        """
        seq_file: A FASTA formatted sequence file
        expression_file: expression file of each enhancer in 17 bins
        TF_file: expression of TFs in 17 bins
        PWM: path to the same PWM file used in GEMSTAT

        nTrain: first nTrain enhancers are used for training
        nValid: next nValid enhancers are used for validation
        nTest: last nTest enhancers used for testing

        nEnhancers: nTrain, nValid and nTest should sum to nEnhancers
        nBins: number of points on V-D axis for which the expression of enhancer was measured
        training: when training is False, all data is scanned with no offset, otherwise only test data is scanned with no offset
        """

        self.nEnhancers_ = nEnhancers
        self.nBins_ = nBins
        # This stores the starting data ID for next batch
        #self.nextBatch_startID = 0


        # has three main keywords: train, valid, and test --> each one is mapped to a second dictionary
        #second disctionary maps data_point_id to another dictionary, third dictionary has following
        #keys:
        #'seq'
        #'seq_encoded'
        #'E_rho'
        #'E_TFs'
        self.data = {}
        self.data['train'] = {}
        self.data['valid'] = {}
        self.data['test'] = {}

        #####################################################################

        tfDict = self.build_TF_dict(PWM)
        scanner = calculate_LLR(seq_file, tfDict)
        scores_dict = scanner.scan(training)
        seqLabels, allSeq = self.read_seq_file(seq_file)
        expr_seqLabels, seqExpr = self.read_expr_file(expression_file)
        tfExpr = self.read_TF_file(TF_file)

        ID_tarin = 0
        ID_valid = 0
        ID_test = 0
        # The first 30 enhancers are used for training, the last 8 are for cross validation
        for si in range(self.nEnhancers_):
            currSeq = allSeq[si]
            currSeqName = seqLabels[si]
            for ofs in scores_dict[currSeqName].keys():
                for bi in range(self.nBins_):

                    ## Check consistency in the input files
                    if seqLabels[si] != expr_seqLabels[si]:
                        #ToDo: handle this case in the code
                        #print seqLabels[si]
                        #print expr_seqLabels[si]
                        raise ValueError('Input files are inconsistent, use the same order for sequences in the input sequence and expression files')

                    """
                    'seq' key is mapped to a python list of size 2 where the first elemnt is sequence and second one is its complement (not reverse complement yet)
                    'seq_encoded' key is mapped to an array containing sequence and its "reverse complement" suitable for conv filters. The second sequence (reverse compl)
                    will be flipped againg after passing to conv filters
                    """
                    if si < nTrain:
                        self.data['train'][ID_tarin] = {}
                        self.data['train'][ID_tarin]['seq'] = currSeq
                        # the below one is a 323*2*3 numpy array consistent with the NN motif implementation
                        self.data['train'][ID_tarin]['seq_score'] = np.stack((scores_dict[currSeqName][ofs]['dl'],scores_dict[currSeqName][ofs]['tw'],scores_dict[currSeqName][ofs]['sn']), axis = 2)
                        self.data['train'][ID_tarin]['E_rho'] = seqExpr[si, bi]
                        self.data['train'][ID_tarin]['E_TFs'] = tfExpr[:,bi]
                        ID_tarin += 1
                    elif si < nTrain + nValid:
                        self.data['valid'][ID_valid] = {}
                        self.data['valid'][ID_valid]['seq'] = currSeq
                        self.data['valid'][ID_valid]['seq_score'] = np.stack((scores_dict[currSeqName][ofs]['dl'],scores_dict[currSeqName][ofs]['tw'],scores_dict[currSeqName][ofs]['sn']), axis = 2)
                        self.data['valid'][ID_valid]['E_rho'] = seqExpr[si, bi]
                        self.data['valid'][ID_valid]['E_TFs'] = tfExpr[:,bi]
                        ID_valid += 1
                    elif ofs == 0: # we do not need various offsets for test data!
                        self.data['test'][ID_test] = {}
                        self.data['test'][ID_test]['seq'] = currSeq
                        self.data['test'][ID_test]['seq_score'] = np.stack((scores_dict[currSeqName][ofs]['dl'],scores_dict[currSeqName][ofs]['tw'],scores_dict[currSeqName][ofs]['sn']), axis = 2)
                        self.data['test'][ID_test]['E_rho'] = seqExpr[si, bi]
                        self.data['test'][ID_test]['E_TFs'] = tfExpr[:,bi]
                        ID_test += 1
        self.nTrain_data = len(self.data['train'].keys())
        if training:
            self.shuffleIDS()
    def build_TF_dict(self, PWM):
        ret = {}
        df = pd.read_csv(PWM, header=None, index_col=None, sep='\t', names = ['0','1','2','3'])
        df = df.values

        for l in df:
            if l[0][0] == '>':
                tfName = l[0][1:]
                ret[tfName] = np.zeros((10,4))
                baseInd = 0
            elif l[0][0] != '<':
                currBase = [float(b) for b in l]
                currBase = np.true_divide(currBase, np.sum(currBase))
                ret[tfName][baseInd,:] = currBase
                baseInd += 1

        return ret

    def shuffleIDS(self):
        ids = np.arange(self.nTrain_data)
        np.random.shuffle(ids)
        self.shuffled_id_ = ids

    def read_seq_file (self, seq_file, length = 332):
        """
        This method reads all the sequences from the sequence file and add 'N' to sequences
        that have a shorter length than the specified length. 'N' will be encoded to
        an all-zero column
        """
        df = pd.read_csv(seq_file, header=None, index_col=None, sep='\n')
        df = df.values
        allSeq = []
        seqLabel = []

        for l in df:
            currLine = l[0]

            if currLine[0] != '>':
                if len(currLine) < length:
                    d = length - len(currLine)
                    for i in range(d):
                        currLine = 'N' + currLine

                    rev_currLine = self.reverse_complement(currLine)
                    allSeq.append([currLine,rev_currLine])

                else:
                    rev_currLine = self.reverse_complement(currLine)
                    allSeq.append([currLine,rev_currLine])
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


    def read_expr_file(self, expr_file):
        df = pd.read_csv(expr_file, header=0, index_col=0, sep='\t')
        label = df.axes[0].values
        expr = df.values

        return label, expr

    def read_TF_file(self, TF_file):
        df = pd.read_csv(TF_file, header=0, index_col=0, sep='\t')
        expr = df.values

        return expr

    def encode_seq(self, seq_string):
        """
        Input: one DNA sequence as string of length L
        returns: np.array(L,4), one-hot encoded sequence

        4 columns: A, C, G, T
        """

        ret = np.zeros((len(seq_string),1,4))
        for ib, b in enumerate(seq_string):
            if b.capitalize() == 'A':
                ret[ib,0, 0] = 1
            elif b.capitalize() == 'C':
                ret[ib,0, 1] = 1
            elif b.capitalize() == 'G':
                ret[ib,0, 2] = 1
            elif b.capitalize() == 'T':
                ret[ib,0, 3] = 1

            # else b == 'N': let it remain zero

        return ret

    def next_batch(self, all_data = None, size = 20):
        """
        all_data : None, 'train', 'test', 'all'
        if all_data == None:
            This method returns a batch of training data of the given size if all_data == None.

        if all_data == 'train':
            returns all of the training data

        if all_data == 'test'
            returns all of the test data

        if all_data == 'valid'
            returns all of the validation data

        if all_data == 'all'
            return all of the training and test data

        Returns in none mode:
            sequence = list(nBatch * np.arrays(batch_size * L * 4))
            TF_concentration = list(nBatch * np.array(batch_size,3))
            rho_expression = list(nBatch * np.array(batch_size,1))

        Returns in all the other modes:
            sequence = np.arrays(data_size * L * 4)
            TF_concentration = np.array(data_size,3)
            rho_expression = np.array(data_size,1)
        """
        sequence = []
        TF_concentration = []
        rho_expression = []

        if all_data == None:
            lenTrainData = len(self.data['train'].keys())
            nBatch = int(np.true_divide(lenTrainData, size))
            idx = 0

            for b in range(nBatch):
                currBatch_seq = []
                currBatch_TF = []
                currBatch_rho = []
                for i in range(size):

                    id_shuffled = self.shuffled_id_[idx]
                    currSeq = self.data['train'][id_shuffled]['seq_score'].tolist()
                    currTF_conc = self.data['train'][id_shuffled]['E_TFs'].tolist()
                    currRho_expr = self.data['train'][id_shuffled]['E_rho'].tolist()

                    currBatch_seq.append(currSeq)
                    currBatch_TF.append(currTF_conc)
                    currBatch_rho.append(currRho_expr)

                    idx += 1
                sequence.append(np.array(currBatch_seq))
                TF_concentration.append(np.array(currBatch_TF))
                rho_expression.append(np.array(currBatch_rho).reshape((size, 1)))

            self.shuffleIDS()
            return sequence, TF_concentration, rho_expression

        elif all_data == 'train':
            lenTrainData = len(self.data['train'].keys())
            print(lenTrainData)
            for ID in range(lenTrainData):
                currSeq = self.data['train'][ID]['seq_score'].tolist()
                currTF_conc = self.data['train'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['train'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)


            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression).reshape((np.shape(rho_expression)[0], 1))

        elif all_data == 'test':
            lenTestData = len(self.data['test'].keys())

            for ID in range(lenTestData):
                currSeq = self.data['test'][ID]['seq_score'].tolist()
                currTF_conc = self.data['test'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['test'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression).reshape((np.shape(rho_expression)[0], 1))

        elif all_data == 'valid':
            lenValidData = len(self.data['valid'].keys())

            for ID in range(lenValidData):
                currSeq = self.data['valid'][ID]['seq_score'].tolist()
                currTF_conc = self.data['valid'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['valid'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression).reshape((np.shape(rho_expression)[0], 1))

        elif all_data == 'all':
            lenTrainData = len(self.data['train'].keys())
            lenTestData = len(self.data['test'].keys())
            lenValidData = len(self.data['valid'].keys())

            for ID in range(lenTrainData):
                currSeq = self.data['train'][ID]['seq_score'].tolist()
                currTF_conc = self.data['train'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['train'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            for ID in range(lenValidData):
                currSeq = self.data['valid'][ID]['seq_score'].tolist()
                currTF_conc = self.data['valid'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['valid'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            for ID in range(lenTestData):
                currSeq = self.data['test'][ID]['seq_score'].tolist()
                currTF_conc = self.data['test'][ID]['E_TFs'].tolist()
                currRho_expr = self.data['test'][ID]['E_rho'].tolist()

                sequence.append(currSeq)
                TF_concentration.append(currTF_conc)
                rho_expression.append(currRho_expr)

            return np.array(sequence), np.array(TF_concentration), np.array(rho_expression).reshape((np.shape(rho_expression)[0], 1))
