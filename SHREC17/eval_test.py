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
                    # continue
                    # _id, synsetId, subSynsetId, modelId, split = line
                    # if _id == "id":
                    #     continue
                    # self.truth[_id] = (synsetId, subSynsetId, split)
                    #
                    # try:
                    #     self.bySynset[synsetId].append(_id)
                    # except:
                    #     self.bySynset[synsetId] = [_id]
                    #
                    # try:
                    #     self.bySubSynset[synsetId].append(_id)
                    # except:
                    #     self.bySubSynset[synsetId] = [_id]
                    #
                    # try:
                    #     self.bySplitSynset[split][synsetId].append(_id)
                    # except:
                    #     self.bySplitSynset[split][synsetId] = [_id]
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
                    print('evaluating',dataset+'...')
                    results[dataset] = self.evaluateRankedLists(datasetDir)
            except:
                continue

        self.saveEvaluationState(results, os.path.abspath(datasetDir),None) # must be only base name of path (just dir?)
        return results

    def results_to_score(self):
        plop

    def zero_pad(self, n, p):
        plop

    def readQueryResults(self, dir):
        plop

        def line_to_query_result(results, line):
            plop

        # itrerate over results

    def saveEvaluationState(self, results, method, allCategoryRes):
        plop

    def evaluateRankedLists(self, dir):

        plop
        return plop

    def writeOracleResults(self):
        plop
