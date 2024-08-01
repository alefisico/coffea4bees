import unittest
#import argparse
from coffea.util import load
import yaml
#from parser import wrapper
import sys

import numpy as np
import awkward as ak
from coffea.nanoevents.methods import vector
import time
from copy import copy
import os

sys.path.insert(0, os.getcwd())
from jet_clustering.clustering   import kt_clustering, cluster_bs, cluster_bs_fast
from jet_clustering.declustering import compute_decluster_variables, decluster_combined_jets, make_synthetic_event, get_list_of_splitting_types, clean_ISR, get_list_of_ISR_splittings, children_jet_flavors

#import vector
#vector.register_awkward()
from coffea.nanoevents.methods.vector import ThreeVector
import fastjet




class clusteringTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(self):
#        self.inputFile = wrapper.args["inputFile"]

        #
        # From 4jet events
        #   (from analysis.helpers.topCandReconstruction import dumpTopCandidateTestVectors
        #
        self.input_jet_pt_4  = [[150.25, 127.8125, 60.875, 46.34375], [169.5, 105.3125, 82.375, 63.0], [174.625, 116.5625, 98.875, 58.96875], [297.75, 152.125, 108.3125, 49.875], [232.0, 176.5, 79.4375, 61.84375], [113.5625, 60.78125, 58.84375, 54.40625], [154.875, 137.75, 102.4375, 64.125], [229.75, 156.875, 109.75, 53.21875], [117.125, 94.75, 57.0, 56.1875], [149.875, 116.8125, 97.75, 42.5]]
        self.input_jet_eta_4 = [[0.708740234375, 2.24365234375, -0.37176513671875, -0.5455322265625], [-0.632080078125, 0.4456787109375, -0.27459716796875, -0.04097747802734375], [-2.03564453125, -0.10174560546875, -0.553466796875, -1.352783203125], [-0.26214599609375, -0.5142822265625, 0.6898193359375, 0.8153076171875], [1.438232421875, -0.66015625, 1.7353515625, 2.0283203125], [0.5064697265625, -1.33837890625, 1.122314453125, 0.31939697265625], [1.6240234375, 0.073394775390625, -2.119140625, -2.2568359375], [2.0361328125, 1.675048828125, 0.95849609375, 0.54150390625], [0.14300537109375, 0.9183349609375, -1.08251953125, 1.379150390625], [0.500244140625, -0.0623931884765625, -0.701416015625, -0.88671875]]
        self.input_jet_phi_4 = [[2.4931640625, -0.48309326171875, 2.66259765625, -1.79443359375], [-0.2913818359375, 2.51220703125, -2.73876953125, 0.58349609375], [-2.220703125, 0.6153564453125, 1.251708984375, -1.930908203125], [-1.36962890625, 1.342041015625, 1.99609375, 2.5849609375], [-0.1124420166015625, 2.6875, -2.44775390625, 0.168304443359375], [2.546875, 1.327392578125, -0.794189453125, -0.979248046875], [2.95556640625, 0.7203369140625, -1.276611328125, -0.4969482421875], [1.421630859375, -1.33935546875, -1.302978515625, -3.140625], [-2.45751953125, 0.27557373046875, 1.65087890625, -0.6121826171875], [-3.08984375, -0.14752197265625, 0.2174072265625, -2.95947265625]]
        self.input_jet_mass_4 = [[16.8125, 24.96875, 9.5390625, 6.18359375], [18.859375, 15.296875, 13.5, 7.7421875], [20.5, 16.96875, 11.7265625, 10.7421875], [20.421875, 16.921875, 16.46875, 9.1875], [32.3125, 18.015625, 10.4140625, 13.40625], [14.046875, 9.625, 12.3984375, 8.3515625], [19.3125, 22.875, 13.671875, 12.0234375], [32.15625, 11.8125, 17.25, 11.3828125], [17.0, 14.953125, 9.046875, 11.5], [15.65625, 16.890625, 17.640625, 7.9921875]]

        self.jet_flavor_4 = [["b"] * 4] * len(self.input_jet_pt_4)

        self.input_jets_4 = ak.zip(
            {
                "pt": self.input_jet_pt_4,
                "eta": self.input_jet_eta_4,
                "phi": self.input_jet_phi_4,
                "mass": self.input_jet_mass_4,
                "jet_flavor": self.jet_flavor_4,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )

        self.input_jet_pt_5  = [[256.330078125, 133.5384521484375, 120.5, 56.39178466796875, 29.992462158203125], [134.58294677734375, 84.63372802734375, 69.49273681640625, 68.5399169921875, 52.375], [235.428466796875, 190.12060546875, 71.2095947265625, 65.34912109375, 60.4375], [205.2244873046875, 107.375, 94.7037353515625, 93.92022705078125, 92.0400390625], [178.065185546875, 120.625, 94.33807373046875, 84.4742431640625, 72.0380859375], [128.86138916015625, 126.4375, 119.819580078125, 60.375732421875, 47.94287109375], [161.7626953125, 74.89837646484375, 71.490478515625, 56.180023193359375, 50.375], [230.293212890625, 136.250244140625, 62.6171875, 58.7060546875, 49.34375], [228.0, 173.1611328125, 96.181640625, 79.3907470703125, 76.552734375], [128.54522705078125, 106.4375, 84.93209838867188, 64.737548828125, 49.700469970703125]]
        self.input_jet_eta_5  = [[0.8936767578125, 0.121337890625, 1.5341796875, -1.989501953125, -0.7362060546875], [0.29376220703125, 0.9923095703125, 1.52783203125, -0.06246185302734375, -1.863525390625], [2.14892578125, -0.574951171875, -1.082763671875, -0.26849365234375, 0.566650390625], [0.25811767578125, -1.43603515625, 1.063720703125, 1.9384765625, 0.408935546875], [0.46978759765625, 1.232666015625, -1.802734375, -1.022705078125, -1.654052734375], [1.38623046875, 0.6514892578125, 1.039794921875, -0.588623046875, 0.49981689453125], [1.182861328125, 0.174346923828125, -0.564208984375, 1.013671875, -1.2578125], [0.45703125, -0.30816650390625, -0.528564453125, 2.0302734375, -0.859375], [-2.0966796875, -0.52490234375, 0.931640625, 0.096832275390625, -0.21209716796875], [0.046417236328125, -0.7169189453125, 1.510986328125, 0.0444183349609375, 0.19891357421875]]
        self.input_jet_phi_5  = [[2.06884765625, 5.5608954429626465, -1.4248046875, 3.7079901695251465, 1.1357421875], [3.7953925132751465, 0.8619384765625, 0.7774658203125, 6.2029852867126465, 3.04150390625], [1.671142578125, 4.9018378257751465, 5.0817694664001465, 5.3899970054626465, 2.7255859375], [5.0261054039001465, 2.37890625, 2.68701171875, 0.7398681640625, 6.0952277183532715], [1.554931640625, -3.13134765625, 5.6200995445251465, 0.2047119140625, 4.4792304039001465], [2.47021484375, 1.626953125, 4.9301581382751465, 6.0857672691345215, 4.9946112632751465], [1.499267578125, 4.5829901695251465, 4.7106757164001465, 5.0483222007751465, -2.0654296875], [4.3429999351501465, 1.481689453125, 0.78515625, 0.6368408203125, 1.269775390625], [-2.8740234375, 1.008544921875, 4.2407050132751465, 5.5678534507751465, 0.562255859375], [1.16943359375, -2.26611328125, 4.9030585289001465, 5.8507513999938965, 1.7421875]]
        self.input_jet_mass_5  = [[18.3929443359375, 16.167137145996094, 20.8125, 9.837577819824219, 7.376430511474609], [18.627456665039062, 14.282325744628906, 10.098735809326172, 10.676605224609375, 12.8828125], [20.226531982421875, 35.16229248046875, 9.027721405029297, 14.4912109375, 9.1015625], [22.6912841796875, 19.15625, 15.039527893066406, 5.409450531005859, 13.238250732421875], [25.096435546875, 14.2734375, 11.000885009765625, 9.951026916503906, 10.315155029296875], [14.715652465820312, 11.21875, 14.8203125, 8.70376968383789, 8.22491455078125], [20.9124755859375, 10.833625793457031, 11.0955810546875, 10.830390930175781, 9.59375], [23.3994140625, 16.117172241210938, 10.51025390625, 10.590438842773438, 7.5078125], [23.140625, 24.85760498046875, 14.499320983886719, 12.903610229492188, 10.6402587890625], [25.775680541992188, 16.75, 12.032047271728516, 13.135284423828125, 6.7713165283203125]]
        self.input_jet_flavor_5  = [['b', 'b', 'j', 'b', 'b'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['b', 'j', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['j', 'b', 'b', 'b', 'b'], ['b', 'j', 'b', 'b', 'b']]

        self.input_jets_5 = ak.zip(
            {
                "pt": self.input_jet_pt_5,
                "eta": self.input_jet_eta_5,
                "phi": self.input_jet_phi_5,
                "mass": self.input_jet_mass_5,
                "jet_flavor": self.input_jet_flavor_5,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )


        self.input_jet_pt_bbj      = [[121.32373046875, 104.4700927734375, 98.575927734375, 76.08642578125, 40.5], [189.64794921875, 103.1923828125, 54.4713134765625, 53.684326171875, 52.28125], [286.233642578125, 160.190185546875, 137.513427734375, 63.8873291015625, 47.0625], [137.4715576171875, 96.1270751953125, 75.84152221679688, 65.9375, 50.4638671875], [323.55615234375, 287.734375, 133.1051025390625, 72.7578125, 44.84375], [212.763427734375, 153.80810546875, 93.18988037109375, 55.65625, 51.38140869140625], [182.375, 159.4368896484375, 117.1884765625, 72.4398193359375, 61.2030029296875], [166.875, 133.4300537109375, 82.14520263671875, 61.18505859375, 55.551177978515625], [188.04296875, 131.243408203125, 84.04052734375, 73.125, 59.80682373046875], [211.049072265625, 124.302978515625, 113.849609375, 96.976318359375, 64.9375]]
        self.input_jet_eta_bbj     = [[-0.27557373046875, -0.7879638671875, 1.0927734375, 1.110107421875, 2.0224609375], [-0.556884765625, 1.50048828125, -1.5009765625, 2.0869140625, 0.9755859375], [0.798095703125, 0.99658203125, -2.26416015625, -0.6783447265625, -0.12933349609375], [1.687744140625, -0.25347900390625, -2.001953125, 1.846923828125, -0.14910888671875], [0.9591064453125, -0.7325439453125, -0.18792724609375, 1.870361328125, 1.698974609375], [1.033203125, -0.6864013671875, 1.393798828125, 1.2431640625, -1.3427734375], [-0.44427490234375, 0.34490966796875, -0.6148681640625, 0.174468994140625, 0.1990966796875], [1.328125, -0.83154296875, -1.185791015625, -0.37445068359375, 1.9853515625], [0.31536865234375, 1.581787109375, 0.690185546875, 0.30084228515625, -1.00927734375], [0.015628814697265625, 1.1064453125, 1.861328125, -0.9864501953125, 2.30322265625]]
        self.input_jet_phi_bbj     = [[0.8409423828125, 0.7689208984375, 4.4438300132751465, 3.3007636070251465, -1.004638671875], [1.4921875, 4.0444159507751465, 3.9799628257751465, 6.166562557220459, -0.5740966796875], [5.7452216148376465, 2.49169921875, 3.1752753257751465, 0.9876708984375, -0.06793212890625], [2.87060546875, 0.0531463623046875, 5.4795966148376465, -0.215606689453125, 2.923828125], [0.38848876953125, 3.3657050132751465, 2.99462890625, 1.03857421875, -1.219970703125], [2.921875, 0.23577880859375, 4.8925604820251465, 0.3216552734375, 2.28955078125], [0.7491455078125, 4.6284003257751465, 2.3427734375, 1.568603515625, 6.0012030601501465], [-2.40283203125, 3.4604315757751465, 0.4329833984375, 0.566650390625, 1.07666015625], [5.6957831382751465, 2.27001953125, 4.5253729820251465, -2.99365234375, 1.842041015625], [1.079345703125, 4.3469061851501465, 4.5226874351501465, 1.9443359375, -2.33837890625]]
        self.input_jet_mass_bbj    = [[17.1162109375, 14.593124389648438, 16.187713623046875, 14.678573608398438, 9.5234375], [42.87664794921875, 15.475616455078125, 10.702194213867188, 9.176605224609375, 9.8515625], [23.61243438720703, 19.530120849609375, 13.258628845214844, 12.957328796386719, 9.90625], [14.27728271484375, 14.1954345703125, 13.277191162109375, 10.46875, 8.628997802734375], [46.4573974609375, 58.197998046875, 28.2264404296875, 11.06298828125, 7.97265625], [18.99920654296875, 10.88543701171875, 13.593544006347656, 7.578125, 8.23480224609375], [23.140625, 19.77544403076172, 28.5819091796875, 9.609878540039062, 9.621047973632812], [15.875, 12.916053771972656, 9.715099334716797, 7.9855499267578125, 10.496315002441406], [22.601318359375, 14.383399963378906, 11.28387451171875, 10.140625, 10.2315673828125], [18.971710205078125, 21.770095825195312, 17.5333251953125, 19.34527587890625, 9.0]]
        self.input_jet_flavor_bbj  = [['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'j', 'b'], ['b', 'b', 'b', 'b', 'j'], ['b', 'b', 'b', 'j', 'b'], ['j', 'b', 'b', 'b', 'b'], ['j', 'b', 'b', 'b', 'b'], ['b', 'b', 'b', 'j', 'b'], ['b', 'b', 'b', 'b', 'j']]


        self.input_jets_bbj = ak.zip(
            {
                "pt": self.input_jet_pt_bbj,
                "eta": self.input_jet_eta_bbj,
                "phi": self.input_jet_phi_bbj,
                "mass": self.input_jet_mass_bbj,
                "jet_flavor": self.input_jet_flavor_bbj,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )


        self.input_jet_pt_6  = [[234.91912841796875, 228.775634765625, 64.3125, 61.700439453125, 58.14398193359375, 48.59375], [246.812744140625, 168.35546875, 131.75, 131.6524658203125, 77.25, 56.489501953125], [456.72607421875, 352.0, 149.59912109375, 86.875, 63.031005859375, 43.9178466796875], [279.784423828125, 250.7904052734375, 101.01605224609375, 60.875, 52.53875732421875, 50.4375], [205.3228759765625, 133.94091796875, 101.066650390625, 92.125, 87.998291015625, 73.75], [188.9853515625, 170.12255859375, 107.0625, 89.625, 72.04833984375, 51.375], [131.75, 106.16046142578125, 83.59130859375, 83.375, 80.9718017578125, 52.386932373046875], [188.0804443359375, 92.1796875, 85.25, 76.4375, 63.67181396484375, 47.8173828125], [223.25, 137.75, 112.379150390625, 81.25457763671875, 78.2041015625, 74.050048828125], [153.71630859375, 137.92950439453125, 72.625, 65.625, 48.224700927734375, 43.3271484375]]
        self.input_jet_eta_6  = [[-0.5133056640625, -1.543701171875, 0.831787109375, 0.9537353515625, -0.7618408203125, -1.153564453125], [0.34356689453125, 0.094818115234375, 0.9638671875, -0.02260589599609375, -0.31683349609375, -1.843505859375], [-1.39208984375, -1.231689453125, -0.9793701171875, -0.11669921875, 0.168609619140625, -0.13671875], [-1.185546875, 1.681396484375, 0.44091796875, 0.45770263671875, -0.3785400390625, -0.6993408203125], [0.06719970703125, -0.7655029296875, -0.005198478698730469, 1.295166015625, 0.669677734375, 2.1767578125], [0.03668975830078125, -0.8272705078125, 1.010009765625, -1.881591796875, 0.592041015625, -1.369384765625], [-0.1100006103515625, -1.47216796875, -0.756591796875, -1.90283203125, -0.605224609375, -0.5802001953125], [0.6260986328125, 1.0166015625, -1.91943359375, -2.1875, 0.588134765625, 0.03516387939453125], [2.29443359375, -2.3876953125, 0.200958251953125, -0.6177978515625, 0.5872802734375, 0.36236572265625], [0.56787109375, -0.610595703125, -2.2841796875, -0.0773773193359375, 0.9136962890625, 0.5615234375]]
        self.input_jet_phi_6  = [[0.49713134765625, 3.7636542320251465, -2.18408203125, 1.37109375, 3.01513671875, 0.0600128173828125], [5.3735175132751465, 2.31298828125, 2.5029296875, 5.6054511070251465, -1.099365234375, 2.41845703125], [1.5263671875, -1.49658203125, 4.2846503257751465, -2.72265625, 0.5770263671875, 0.21160888671875], [1.529296875, 4.7824530601501465, 5.1801581382751465, -2.49658203125, 1.103515625, 2.10205078125], [4.9992499351501465, 1.5009765625, 4.4350409507751465, 2.3486328125, 1.9658203125, 1.104248046875], [5.7166571617126465, 2.6904296875, -0.45654296875, 2.40087890625, 5.1591620445251465, 1.868896484375], [-2.20703125, 1.962158203125, 0.53076171875, -0.107879638671875, 2.99072265625, 6.280444622039795], [3.6713690757751465, 2.3046875, 0.2967529296875, -0.5640869140625, 0.604248046875, 6.1497015953063965], [-1.74560546875, 2.49755859375, 1.0810546875, 1.872802734375, 1.3837890625, 5.2116522789001465], [0.30316162109375, 3.1650214195251465, -2.75439453125, 0.294921875, 5.3921942710876465, 4.9733710289001465]]
        self.input_jet_mass_6  = [[30.531005859375, 29.481903076171875, 9.0234375, 11.388397216796875, 9.933212280273438, 9.6328125], [24.9578857421875, 25.6658935546875, 25.484375, 11.848068237304688, 9.796875, 8.627471923828125], [67.39971923828125, 32.09375, 25.495628356933594, 12.6328125, 9.835433959960938, 8.893363952636719], [37.81341552734375, 27.132110595703125, 21.957839965820312, 11.3828125, 7.413169860839844, 10.8203125], [20.790863037109375, 16.518096923828125, 16.962127685546875, 15.4921875, 12.61669921875, 13.7578125], [29.63092041015625, 26.920989990234375, 17.078125, 21.515625, 10.68695068359375, 10.27667236328125], [21.046875, 15.8487548828125, 11.893730163574219, 17.375, 16.7896728515625, 9.014328002929688], [18.280426025390625, 9.745559692382812, 13.5078125, 22.375, 11.146934509277344, 7.6776123046875], [13.953125, 14.125, 18.94012451171875, 14.496162414550781, 11.415924072265625, 9.816543579101562], [26.37384033203125, 13.808853149414062, 10.6328125, 11.40625, 8.66156005859375, 8.05224609375]]
        self.input_jet_flavor_6  = [['b', 'b', 'j', 'b', 'b', 'j'], ['b', 'b', 'j', 'b', 'j', 'b'], ['b', 'j', 'b', 'j', 'b', 'b'], ['b', 'b', 'b', 'j', 'b', 'j'], ['b', 'b', 'b', 'j', 'b', 'j'], ['b', 'b', 'j', 'j', 'b', 'b'], ['j', 'b', 'b', 'j', 'b', 'b'], ['b', 'b', 'j', 'j', 'b', 'b'], ['j', 'j', 'b', 'b', 'b', 'b'], ['b', 'b', 'j', 'j', 'b', 'b']]


        self.input_jets_6 = ak.zip(
            {
                "pt": self.input_jet_pt_6,
                "eta": self.input_jet_eta_6,
                "phi": self.input_jet_phi_6,
                "mass": self.input_jet_mass_6,
                "jet_flavor": self.input_jet_flavor_6,
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=vector.behavior,
        )



        self.debug = False


    def test_kt_clustering_4jets(self):

        R = np.pi  # Jet size parameter
        clustered_jets = kt_clustering(self.input_jets_4, R)


        jetdefAll = fastjet.JetDefinition(fastjet.kt_algorithm, R)
        clusterAll = fastjet.ClusterSequence(self.input_jets_4, jetdefAll)

        for iEvent, jets in enumerate(clustered_jets):
            if self.debug: print(f"Event {iEvent}")
            for i, jet in enumerate(jets):

                hasFJMatch = False
                if self.debug: print(f"Jet {i+1}: px = {jet.px:.2f}, py = {jet.py:.2f}, pz = {jet.pz:.2f}, E = {jet.E:.2f}, type = {jet.jet_flavor}")
                for i_fj, jet_fj in enumerate(clusterAll.inclusive_jets()[iEvent]):
                    if np.allclose( (jet.px, jet.py, jet.pz, jet.E),(jet_fj.px, jet_fj.py, jet_fj.pz, jet_fj.E), atol=1e-3 ):
                        if self.debug: print("Has match!")
                        hasFJMatch =True

                self.assertTrue(hasFJMatch, " Not all jets have a fastjet match")

            if self.debug:
                for i_fj, jet_fj in enumerate(clusterAll.inclusive_jets()[iEvent]):
                    print(f"FJ  {i_fj+1}: px = {jet_fj.px:.2f}, py = {jet_fj.py:.2f}, pz = {jet_fj.pz:.2f}, E = {jet_fj.E:.2f}")


    def _declustering_test(self, input_jets, debug=False):

        clustered_jets, clustered_splittings = cluster_bs(input_jets, debug=False)
        compute_decluster_variables(clustered_splittings)

        if self.debug:
            for iEvent, jets in enumerate(clustered_jets):
                print(f"Event {iEvent}")
                for i, jet in enumerate(jets):
                    print(f"Jet {i+1}: px = {jet.px:.2f}, py = {jet.py:.2f}, pz = {jet.pz:.2f}, E = {jet.E:.2f}, type = {jet.jet_flavor}")
                print("...Splittings")

                for i, splitting in enumerate(clustered_splittings[iEvent]):
                    print(f"Split {i+1}: px = {splitting.px:.2f}, py = {splitting.py:.2f}, pz = {splitting.pz:.2f}, E = {splitting.E:.2f}, type = {splitting.jet_flavor}")
                    print(f"\tPart_A {splitting.part_A}")
                    print(f"\tPart_B {splitting.part_B}")


        #
        # Declustering
        #

        # Eventually will
        #   Lookup thetaA, Z, mA, and mB
        #   radom draw phi  (np.random.uniform(-np.pi, np.pi, len()) ? )
        #
        #  For now use known inputs
        #   (should get exact jets back!)
        clustered_splittings["decluster_mask"] = True
        pA, pB = decluster_combined_jets(clustered_splittings)


        #
        # Check Pts
        #
        pt_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        if not all(pt_check):
            [print(i) for i in clustered_splittings.pt - (pA + pB).pt]
            [print(i, j) for i, j in zip(clustered_splittings.pt, (pA + pB).pt)]
        self.assertTrue(all(pt_check), "All pt should be the same")

        #
        # Check Eta
        #
        eta_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.eta, (pA + pB).eta)]
        if not all(eta_check):
            [print(i) for i in clustered_splittings.eta - (pA + pB).eta]
            [print(i, j) for i, j in zip(clustered_splittings.eta, (pA + pB).eta)]
        self.assertTrue(all(eta_check), "All eta should be the same")



        #
        # Check Masses
        #
        mass_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        if not all(mass_check):
            [print(i) for i in clustered_splittings.mass - (pA + pB).mass]
            [print(i, j) for i, j in zip(clustered_splittings.mass, (pA + pB).mass)]
        self.assertTrue(all(mass_check), "All Masses should be the same")

        #
        # Check Phis
        #
        phi_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.phi, (pA + pB).phi)]
        if not all(phi_check):
            [print(i) for i in clustered_splittings.phi - (pA + pB).phi]
            [print(i, j) for i, j in zip(clustered_splittings.phi, (pA + pB).phi)]
        self.assertTrue(all(phi_check), "All phis should be the same")


    def test_declustering_4jets(self):
        self._declustering_test(self.input_jets_4)

    def test_declustering_5jets(self):
        self._declustering_test(self.input_jets_5, debug=False)

    def test_declustering_bbjjets(self):
        self._declustering_test(self.input_jets_bbj, debug=False)

    def test_declustering_6jets(self):
        self._declustering_test(self.input_jets_6, debug=False)



    def test_cluster_bs_fast_4jets(self):

        start = time.perf_counter()
        clustered_jets_fast, clustered_splittings_fast = cluster_bs_fast(self.input_jets_4, debug=False)
        end = time.perf_counter()
        elapsed_time_matrix_python = (end - start)
        print(f"\nElapsed time fast Python = {elapsed_time_matrix_python}s")

        start = time.perf_counter()
        clustered_jets, clustered_splittings = cluster_bs(self.input_jets_4, debug=False)
        end = time.perf_counter()
        elapsed_time_loops_python = (end - start)
        print(f"\nElapsed time loops Python = {elapsed_time_loops_python}s")

        #
        # Sanity checks
        #

        #
        # Check Masses
        #
        mass_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.mass, clustered_splittings_fast.mass)]
        if not all(mass_check):
            print("deltas")
            [print(i) for i in clustered_splittings.mass - clustered_splittings_fast.mass]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.mass, clustered_splittings_fast.mass)]
        self.assertTrue(all(mass_check), "All Masses should be the same")


        #
        # Check Pts
        #
        pt_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.pt, clustered_splittings_fast.pt)]
        if not all(pt_check):
            print("deltas")
            [print(i) for i in clustered_splittings.pt - clustered_splittings_fast.pt]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.pt, clustered_splittings_fast.pt)]
        self.assertTrue(all(pt_check), "All Masses should be the same")


        #
        # Check phis
        #
        phi_check = [np.allclose(i, j, 1e-4) for i, j in zip(clustered_splittings.phi, clustered_splittings_fast.phi)]
        if not all(phi_check):
            print("deltas")
            [print(i) for i in clustered_splittings.phi - clustered_splittings_fast.phi]
            print("values")
            [print(i, j) for i, j in zip(clustered_splittings.phi, clustered_splittings_fast.phi)]
        self.assertTrue(all(phi_check), "All phis should be the same")


    def _check_jet_flavors(self, part_A_jet_flavor, part_B_jet_flavor, debug=False):

        #
        # Check particle orderings of splittings
        #
        part_A_flat = ak.flatten(part_A_jet_flavor)
        part_B_flat = ak.flatten(part_B_jet_flavor)

        if debug:
            [print(i) for i in zip(part_A_flat, part_B_flat)]

        #
        #  A should always be the more complex
        #
        part_A_len = np.array([len(i) for i in part_A_flat])
        part_B_len = np.array([len(i) for i in part_B_flat])
        self.assertTrue(all(a >= b for a, b in zip(part_A_len, part_B_len)), "Part A should be the more complex of the pair")


        #
        #  More bs in A when lengths are the same
        #
        part_A_countbs = np.array([i.count("b") for i in part_A_flat])
        part_B_countbs = np.array([i.count("b") for i in part_B_flat])

        more_bs_in_partA = part_A_countbs >= part_B_countbs
        equal_len_mask = part_A_len == part_B_len
        more_bs_in_partA[~equal_len_mask] = True


        self.assertTrue(np.all(more_bs_in_partA), "Part A should alwasy have more bs")


    def _synthetic_datasets_test(self, input_jets, n_jets_expected, debug=False):

        clustered_jets, _clustered_splittings = cluster_bs(input_jets, debug=False)

        self._check_jet_flavors(_clustered_splittings.part_A.jet_flavor,
                                _clustered_splittings.part_B.jet_flavor,
                                debug=debug)


        #
        #  Decluster the splitting that are 0b + >1 bs
        #
        if debug:
            print("Jet flavour Before ISR cleaning")
            [print(i) for i in clustered_jets.jet_flavor]

        clustered_jets = clean_ISR(clustered_jets, _clustered_splittings, debug=debug)

        if debug:
            print("Jet flavour after ISR cleaning")
            [print(i) for i in clustered_jets.jet_flavor]



        #
        # Declustering
        #

        #
        #  Read in the pdfs
        #
        #  Make with ../.ci-workflows/synthetic-dataset-plot-job.sh
        # input_pdf_file_name = "analysis/plots_synthetic_datasets/clustering_pdfs.yml"
        input_pdf_file_name = "jet_clustering/jet-splitting-PDFs-00-02-00/clustering_pdfs_vs_pT.yml"
        #input_pdf_file_name = "jet_clustering/clustering_PDFs/clustering_pdfs_vs_pT.yml"
        with open(input_pdf_file_name, 'r') as input_file:
            input_pdfs = yaml.safe_load(input_file)

        declustered_jets = make_synthetic_event(clustered_jets, input_pdfs, debug=debug)
        #pA = declustered_jets[:,0:2]
        #pB = declustered_jets[:,2:]

        #
        # Sanity checks
        #


        match_n_jets = ak.num(declustered_jets) == n_jets_expected
        if not all(match_n_jets):
            print("ERROR number of declustered_jets")
            print(f"ak.num(declustered_jets)        {ak.num(declustered_jets)}")
            print(f"clustered_jets.jet_flavor     {clustered_jets.jet_flavor}")
            print(f"clustered_jets.pt             {clustered_jets.pt}")
            #print(f"pA.pt                         {pA.pt}")
            #print(f"pB.pt                         {pB.pt}")

            print(f"declustered_jets.pt             {declustered_jets.pt}")
            print(f"ak.num(declustered_jets)        {ak.num(declustered_jets)}")
            print(f"clustered_jets.phi             {clustered_jets.phi}")

            #
            #  Checkphi
            #
            #print(f"input phi {clustered_jets.phi[1]}")
            #print(f"Reco phi {(pA + pB).phi[1]}")

        self.assertTrue(all(match_n_jets), f"Should always get {n_jets_expected} jets")

        #
        #  Do reclustering
        #
        is_b_mask = declustered_jets.jet_flavor == "b"

        canJet = declustered_jets[is_b_mask]
        notCanJet_sel = declustered_jets[~is_b_mask]
        jets_for_clustering = ak.concatenate([canJet, notCanJet_sel], axis=1)
        jets_for_clustering = jets_for_clustering[ak.argsort(jets_for_clustering.pt, axis=1, ascending=False)]
        clustered_jets_reclustered, clustered_splittings_reclustered = cluster_bs(jets_for_clustering, debug=False)
        compute_decluster_variables(clustered_splittings_reclustered)

        self._check_jet_flavors(clustered_splittings_reclustered.part_A.jet_flavor,
                                clustered_splittings_reclustered.part_B.jet_flavor)


    def test_synthetic_datasets_4jets(self):
        self._synthetic_datasets_test(self.input_jets_4, n_jets_expected = 4)


    def test_synthetic_datasets_5jets(self):
        self._synthetic_datasets_test(self.input_jets_5, n_jets_expected = 5, debug=True)


    def test_synthetic_datasets_bbjjets(self):
        self._synthetic_datasets_test(self.input_jets_bbj, n_jets_expected = 5)


#    def test_synthetic_datasets_6jets(self):
#        self._synthetic_datasets_test(self.input_jets_6, n_jets_expected = 6, debug = True)


    def test_children_jet_flavors(self):
        splitting_types = [ ('bb', ('b','b')), ('bj',('b','j')), ('jb',('j','b')), ('jj',('j','j')),
                            ("j(bb)", ('(bb)', 'j')), ("b(bj)", ('(bj)', 'b')), ("j(bj)", ('(bj)', 'j')), ("(bj)b", ('(bj)', 'b')),
                            ("(j(bj))b", ('(j(bj))', 'b')), ("(bb)(jj)", ('(bb)', '(jj)')), ("(jj)(bb)", ('(jj)', '(bb)')), ("j(j(bj))", ('(j(bj))','j')),
                           ]

        for _s in splitting_types:

            _children = children_jet_flavors(_s[0])
            #print(_s[0],":",_children,"vs",_s[1])
            self.assertTrue(_children == _s[1], f"Miss match for type {_s[0]}: got {_children}, expected {_s[1]}")

    def test_get_list_of_ISR_splittings(self):


        splitting_types = [("b",False), ("j",False),
                           ("bb",False), ("bj",True), ("jj",True),
                           ("j(bb)",True), ("b(bj)",False), ("j(bj)",True),('(bj)b',False),
                           ("b(j(bj))",False), ('(bb)(jj)',True), ('(jj)(bb)',True) ,("j(j(bj))",True), ("j(b(bj))",True)
                           ]


        test_splitting_types = []
        expected_ISR_splittings = []

        for _s in splitting_types:
            test_splitting_types.append(_s[0])
            if _s[1]:
                expected_ISR_splittings.append(_s[0])


        ISR_splittings = get_list_of_ISR_splittings(test_splitting_types)

        print(f"ISR_splittings is {ISR_splittings}")

        self.assertListEqual(ISR_splittings, expected_ISR_splittings)


if __name__ == '__main__':
    # wrapper.parse_args()
    # unittest.main(argv=sys.argv)
    unittest.main()
