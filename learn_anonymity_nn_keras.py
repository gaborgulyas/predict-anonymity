from igraph import *
from scipy.stats import spearmanr
import numpy as np
import pickle
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def load_data(results_path, graph_path):
	# Prepare graph
	g = Graph.Read_Ncol(graph_path, names=True, weights=False, directed=False)
	for v in g.vs:
		v['id'] = int(v['name'])
	del g.vs['name']

	# Load results of the simulations
	facc = open(results_path)
	lines = facc.readlines()
	common_ids_tmp = lines[0].replace("\n", "").replace("\r", "").split(";")
	common_ids = [int(x) for x in common_ids_tmp if x != ""]
	scores_tmp = lines[1].replace("\n", "").replace("\r", "").split(";")
	scores = [int(x) for x in scores_tmp if x != ""]
	facc.close()

	# print float(scores.count(0)) / float(len(scores))

	# Count node re-identification scores
	indexed_scores = {}
	id_to_gvix = {}
	for v in g.vs:
		if v['id'] not in common_ids or g.degree(v.index) > 250:
			continue
		indexed_scores[v['id']] = scores[common_ids.index(v['id'])]
		id_to_gvix[v['id']] = v.index

	max_deg = float(max(g.degree()))

	# Make the graph data to a list of node fingerprints
	graph_data = []
	for vid in sorted(indexed_scores.keys()):
		vix = id_to_gvix[vid]
		data_entry = []
		for nbr in g.neighbors(vix):
			data_entry.append(g.degree(nbr))
		data_entry = sorted(data_entry)
		data_entry = [g.degree(vix)] + [0] * (250 - len(data_entry)) + data_entry
		# data_entry2 = [float(x)/max_deg for x in data_entry]
		data_entry2 = [min(float(x)/250, 1.0) for x in data_entry]
		graph_data.append(data_entry2)

	graph_structure_data = np.array(graph_data)

	node_ids = sorted(indexed_scores.keys())
	scores = np_utils.to_categorical([indexed_scores[vid]+10 for vid in sorted(indexed_scores.keys())], 21)
	return node_ids, graph_structure_data, scores

networks = ["$NW$_v=0.25_e=0.25", "$NW$_v=0.25_e=0.5", "$NW$_v=0.25_e=0.75", "$NW$_v=0.25_e=1.0", "$NW$_v=0.5_e=0.25", "$NW$_v=0.5_e=0.5", "$NW$_v=0.5_e=0.75", "$NW$_v=0.5_e=1.0", "$NW$_v=0.75_e=0.25", "$NW$_v=0.75_e=0.5", "$NW$_v=0.75_e=0.75", "$NW$_v=0.75_e=1.0", "$NW$_v=1.0_e=0.25", "$NW$_v=1.0_e=0.5"]
# skipped: "$NW$_v=1.0_e=0.75", "$NW$_v=1.0_e=1.0"
train_nw = "slashdot"
test_nw = "epinions"

#####
###
#	PREDICT
###
#####

correlation_values = {}

use_cache = True

for nw in networks:
	print "$", nw

	print "\tCreating NEURAL MODEL...",
	model = Sequential()
	model.add(Dense(251, 128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, 128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(128, 21, activation='softmax'))

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	# rms = RMSprop()
	model.compile(loss='mean_squared_error', optimizer=sgd)
	print "OK"

	#####
	#	TRAIN phase
	#####
	## Load data
	print "\tLoading TRAINING DATA...",
	if not use_cache:
		node_ids, train_data, score = load_data("./SimuData/" + nw.replace('$NW$', train_nw) + "/e0_v0_sum.csv",
												"./SimuData/" + nw.replace('$NW$', train_nw) + "/e0_v0_src.tgf")
		pickle.dump(node_ids, open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_node_ids.p", "w+"))
		pickle.dump(train_data, open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_train_data.p", "w+"))
		pickle.dump(score, open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_score.p", "w+"))
	else:
		node_ids = pickle.load(open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_node_ids.p", "r"))
		train_data = pickle.load(open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_train_data.p", "r"))
		score = pickle.load(open("./cache/"+nw.replace('$NW$', train_nw)+"_e0_v0_score.p", "r"))
	print "OK"

	## Train the random forest
	print "\tTRAINING network is in progress...",
	model.fit(train_data, score, nb_epoch=20, batch_size=32, show_accuracy=True, verbose=2)
	print "OK"

	#####
	#	PREDICT phase
	#####
	## Load the expected TARGET data
	print "\tLoading TEST DATA ("+nw+")...",
	if not use_cache:
		node_ids, test_data, expected_target_data = load_data("./SimuData/" + nw.replace('$NW$', test_nw) + "/e0_v0_sum.csv",
															  "./SimuData/" + nw.replace('$NW$', test_nw) + "/e0_v0_src.tgf")
		pickle.dump(node_ids, open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_node_ids.p", "w+"))
		pickle.dump(test_data, open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_test_data.p", "w+"))
		pickle.dump(expected_target_data, open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_expected_score.p", "w+"))
	else:
		node_ids = pickle.load(open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_node_ids.p", "r"))
		test_data = pickle.load(open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_test_data.p", "r"))
		expected_target_data = pickle.load(open("./cache/"+nw.replace('$NW$', test_nw)+"_e0_v0_expected_score.p", "r"))
	print "OK"

	# Take the same decision trees and run it on the test data
	print "\tMaking PREDICTIONS...",
	eval_score = model.evaluate(test_data, expected_target_data, batch_size=32, show_accuracy=True, verbose=0)
	print 'Test score:', eval_score[0]
	print 'Test accuracy:', eval_score[1]
	predicted_data = model.predict(test_data, batch_size=32)
	print "OK"

	# Measure the quality of prediction
	print "\tSaving PREDICTIONS to CSV file...",
	x = []
	y = []
	fout = open("prediction_results_" + nw + ".csv", "w+")
	fout.write("ID;re-id_count;predicted\n")
	err_ctr = 0
	for ix in range(len(node_ids)):
		id = node_ids[ix]

		expected_value = expected_target_data[ix].tolist().index(1)-10

		line = str(id) + ";"
		line += str(expected_value) + ";"

		prediction = predicted_data[ix].tolist()
		predicted_value = prediction.index(max(prediction))-10

		line += str(predicted_value) + ";"
		fout.write(line + "\n")
		x.append(expected_value)
		y.append(predicted_value)
	fout.close()
	print "OK"

	print "\tCalculating RANK CORRELATION...",
	sc = spearmanr(x, y)
	print "OK"
	print "\tSpearman correlation: ", sc[0]
	correlation_values[nw] = sc[0]

	print ""

for nw in sorted(correlation_values.keys()):
	print nw, correlation_values[nw]
