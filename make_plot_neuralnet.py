import numpy as np
import matplotlib.pyplot as plt
import sys

recalls = [1.13109856993, 5.16098605886, 10.9927308692, 14.8603846265, 0.941676094161, 17.2142729196, 25.9405351276, 32.6994225522, 0.62430822548, 25.3924998931, 36.1381906328, 44.4267804704, 0.439566632977]

lta_a = [-0.6648386340467163, -0.6595399472541462, -0.6542197707884314, -0.6449867876285473, -0.6182122995009147, -0.6284719457480625, -0.6280234273514589, -0.630811768411019, -0.552650970882677, -0.6122324313027292, -0.6338551552674494, -0.6313783286431212, -0.4696999572398811]
lta_a = [abs(d) for d in lta_a]

deg = [0.7065206664660384, 0.6936129901178901, 0.7007253333673702, 0.6928460697885981, 0.6580906336852326, 0.6608683924729697, 0.6615129045996133, 0.6686085123509411, 0.5850230964724288, 0.6302926188673572, 0.6565477943571505, 0.6727273095618249, 0.49570275674634356]

ml_rf = [0.840814692001, 0.852665649519, 0.792802398236, 0.75537390916, 0.840255946369, 0.717316060758, 0.654104129153, 0.635768170409, 0.654275278909, 0.624238255844, 0.6251574844, 0.614694528139, 0.455267014016]
ml_nn = [0.777888806688, 0.77499364952, 0.739628449613, 0.73779799719, 0.760399211579, 0.722980096821, 0.697638017541, 0.672254554893, 0.662072392878, 0.677357462216, 0.653447943061, 0.64734271701, 0.209908314886, 0.650751792576]

x_marks = [x for x in range(0, 51, 10)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0.0, 50.0)
ax.set_xticks(ticks=x_marks)
ax.set_ylim(0.4, 0.9)
ax.tick_params('both', labelsize=14)
ax.set_ylabel("$|\\rho_S(S(v), LTA_i(v))|$", fontsize=20)

zero = [0.0 for x in x_marks]
zero_plot, = ax.plot(x_marks, zero, "r--")

for i in range(0, len(recalls)):
	r1, = ax.plot(recalls[i], lta_a[i], "bs")
	r2, = ax.plot(recalls[i], deg[i], "r*")
	r3, = ax.plot(recalls[i], ml_rf[i], "go")
	r4, = ax.plot(recalls[i], ml_nn[i], "y^")

ax.legend([r1, r2, r3, r4], ["$LTA_A$", "$LTA_{deg}$", "RandomF", "NeuralN"], loc="lower center", ncol=2, fontsize=16)

plt.xlabel("Recall rates (%)", fontsize=19)
plt.savefig("plots/correlations_neuralnet.png", dpi=100, bbox_inches='tight')
plt.savefig("plots/correlations_neuralnet.pdf", bbox_inches='tight')
