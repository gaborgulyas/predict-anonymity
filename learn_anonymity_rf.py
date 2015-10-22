from igraph import *
from sklearn.ensemble import RandomForestClassifier
#from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr

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
	target_data = [int(x) for x in scores_tmp if x != ""]
	facc.close()

	# Count node re-identification scores
	indexed_scores = {}
	id_to_index = {}
	for v in g.vs:
		if v['id'] not in common_ids or g.degree(v.index) > 250:
			continue
		indexed_scores[v['id']] = target_data[common_ids.index(v['id'])]
		id_to_index[v['id']] = v.index

	# Make the graph data to a list of node fingerprints
	graph_data = []
	for vid in sorted(indexed_scores.keys()):
		vix = id_to_index[vid]
		data_entry = []
		for nbr in g.neighbors(vix):
			data_entry.append(g.degree(nbr))
		data_entry = sorted(data_entry)
		data_entry = [g.degree(vix)] + [0]*(250 - len(data_entry)) + data_entry
		graph_data.append(data_entry)

	node_ids = sorted(indexed_scores.keys())
	scores = [indexed_scores[vid] for vid in sorted(indexed_scores.keys())]
	return node_ids, graph_data, scores

# networks = ["epinions_v=0.5_e=0.5"]
networks = ["epinions_v=0.25_e=0.25", "epinions_v=0.25_e=0.5", "epinions_v=0.25_e=0.75", "epinions_v=0.25_e=1.0", "epinions_v=0.5_e=0.25", "epinions_v=0.5_e=0.5", "epinions_v=0.5_e=0.75", "epinions_v=0.5_e=1.0", "epinions_v=0.75_e=0.25", "epinions_v=0.75_e=0.5", "epinions_v=0.75_e=0.75", "epinions_v=0.75_e=1.0", "epinions_v=1.0_e=0.25", "epinions_v=1.0_e=0.5"]
# Skipped, due to the lack of memory in python: "epinions_v=1.0_e=0.75", "epinions_v=1.0_e=1.0"

#####
###
#	PREDICT
###
#####

correlation_values = {}

for nw in networks:
	print "$", nw

	# Create the random forest object
	forest = RandomForestClassifier(n_estimators = 100)

	#####
	#	TRAIN phase
	#####
	## Load data
	node_ids, train_data, score = load_data("./SimuData/"+nw.replace('epinions', 'slashdot')+"/e0_v0_sum.csv", "./SimuData/"+nw.replace('epinions', 'slashdot')+"/e0_v0_src.tgf")
	print "TRAINING DATA loaded"

	## Train the random forest
	forest = forest.fit(train_data, score)
	print "TRAINING complete"

	#####
	#	PREDICT phase
	#####
	## Load the expected TARGET data
	node_ids, test_data, expected_target_data = load_data("./SimuData/"+nw+"/e0_v0_sum.csv", "./SimuData/"+nw+"/e0_v0_src.tgf")
	print "TEST DATA loaded >>", nw

	# Take the same decision trees and run it on the test data
	predicted_data = forest.predict(test_data)
	print "PREDICTION done"

	# Measure the quality of prediction
	x = []
	y = []
	fout = open("prediction_results_"+nw+".csv", "w+")
	fout.write("ID;re-id_count;predicted\n")
	for ix in range(len(node_ids)):
		id = node_ids[ix]
		line = str(id) + ";"
		line += str(expected_target_data[ix]) + ";"
		line += str(predicted_data[ix]) + ";"
		fout.write(line + "\n")
		x.append(expected_target_data[ix])
		y.append(predicted_data[ix])
	fout.close()

	sc = spearmanr(x, y)
	print "Spearman correlation: ", sc[0]
	correlation_values[nw] = sc[0]

	print ""

for nw in sorted(correlation_values.keys()):
	print nw, correlation_values[nw]