#!/usr/bin/env python3
# coding: utf-8

"""
    Evaluate the results of the given SHREC17 dataset output
"""

import os
import csv
import gzip
import io

class Evaluator(object):

    def __init__(self, truthpath):
        self.truthfile = truthpath  # must contain all.csv.gz
        self.thruth = {}        # id: synset, sunsynset, split (train, val, test)
        self.bySynset = {}      # synset: [id]
        self.bySubSynset = {}   # subsynset: [id]
        self.bySplitSynset = {'train': {}, 'val': {}, 'test': {}}

        if self.truthfile.endswith('.gz'):
            "dezip"
            with io.BufferedReader(gzip.open(self.truthfile, 'rb')) as file:
                reader = csv.reader(io.TextIOWrapper(file, newline=''))
                for line in reader:
                    self._read_truthfile(line)
        else:
            with open(self.truthfile, 'rb') as file:        # search for io binary with csv
                reader = csv.reader(io.TextIOWrapper(file, newline=''))
                for line in reader:
                    self._read_truthfile(line)

        print("Loaded evaluator using: "+self.truthfile)

    def _read_truthfile(self, line):
        _id, synsetId, subSynsetId, modelId, split = line
        if _id == "id":
            return
        self.truth[_id] = (synsetId, subSynsetId, split)

        try:
            self.bySynset[synsetId].append(_id)
        except:
            self.bySynset[synsetId] = [_id]

        try:
            self.bySubSynset[synsetId].append(_id)
        except:
            self.bySubSynset[synsetId] = [_id]

        try:
            self.bySplitSynset[split][synsetId].append(_id)
        except:
            self.bySplitSynset[split][synsetId] = [_id]

    def evaluate(self, dir):
        datasetNames = ['test_normal', 'test_perturbed',
                        'train_normal', 'train_perturbed',
                        'val_normal', 'val_perrturbed']
        # compute results
        results = {}
        for dataset in datasetNames:
            datasetDir = os.path.join(dir, dataset)
            try:
                if os.path.isdir(datasetDir):
                    print('evaluating', dataset+'...')
                    results[dataset] = self.evaluateRankedLists(datasetDir)
            except:
                continue

        self.saveEvaluationState(results, dir, None) # must be only base name of path (just dir?)
        return results

    def results_to_score(self):
        plop

    def zero_pad(self, n, p):
        for i in range(p-len(n)):
            n = "0"+n
        return n

    def readQueryResults(self, dir):
        import glob
        files = glob.glob(os.path.join(dir, "*.*"))
        allResults = {}

        def line_to_query_result(results, line):
            if(line):
                _id = line[0]
                if len(_id) is not 6:
                    _id = self.zero_pad(_id, 6)
                results.append(_id)
        # itrerate over results
        for file in files:
            results = []
            line2result = line_to_query_result(results, )

    def saveEvaluationState(self, results, method, allCategoryRes):
        plop

    def evaluateRankedLists(self, dir):
        split = os.path.split(dir)[-1].split('_')[0]
        self.readQueryResults(dir)
        plop
        return plop

    def writeOracleResults(self):
        plop
