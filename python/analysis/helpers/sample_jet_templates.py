import numpy as np
import awkward as ak

def sample_PDFs(input_jets_decluster, input_pdfs, splittings):

    n_jets   = np.sum(ak.num(input_jets_decluster))

    #
    #  Sample the PDFs for the jets we will uncluster
    #
    for _var_name in input_pdfs["varNames"]:

        if _var_name.find("_vs_") == -1:
            is_1d_pdf = True
            _sampled_data = np.ones(n_jets)
        else:
            is_1d_pdf = False
            _sampled_data_x = np.ones(n_jets)
            _sampled_data_y = np.ones(n_jets)

        # Sample the pdfs from the different splitting options
        for _splitting_name, _num_samples, _indicies_tuple in splittings:

            if is_1d_pdf:
                probs   = np.array(input_pdfs[_splitting_name][_var_name]["probs"], dtype=float)
                centers = np.array(input_pdfs[_splitting_name][_var_name]["bin_centers"], dtype=float)
                _sampled_data[_indicies_tuple] = np.random.choice(centers, size=_num_samples, p=probs)
            else:
                probabilities_flat   = np.array(input_pdfs[_splitting_name][_var_name]["probabilities_flat"], dtype=float)
                xcenters        = np.array(input_pdfs[_splitting_name][_var_name]["xcenters"],      dtype=float)
                ycenters        = np.array(input_pdfs[_splitting_name][_var_name]["ycenters"],      dtype=float)

                xcenters_flat = np.repeat(xcenters, len(ycenters))
                ycenters_flat = np.tile(ycenters, len(xcenters))

                sampled_indices = np.random.choice(len(probabilities_flat), size=_num_samples, p=probabilities_flat)

                _sampled_data_x[_indicies_tuple] = xcenters_flat[sampled_indices]
                _sampled_data_y[_indicies_tuple] = ycenters_flat[sampled_indices]

        #
        # Save the sampled data to the jets to be uclustered for use in decluster_combined_jets
        #
        if is_1d_pdf:
            input_jets_decluster[_var_name]         = ak.unflatten(_sampled_data,    ak.num(input_jets_decluster))
        else:
            input_jets_decluster["zA"]         = ak.unflatten(_sampled_data_x,    ak.num(input_jets_decluster))
            input_jets_decluster["thetaA"]     = ak.unflatten(_sampled_data_y,    ak.num(input_jets_decluster))



def sample_PDFs_vs_pT(input_jets_decluster, input_pdfs, splittings):

    n_jets   = np.sum(ak.num(input_jets_decluster))

    n_pt_bins = len(input_pdfs["pt_bins"]) - 1
    pt_masks = []
    for iPt in range(n_pt_bins):
        _min_pt = float(input_pdfs["pt_bins"][iPt])
        _max_pt = float(input_pdfs["pt_bins"][iPt+1])
        if _max_pt == "inf":
            _max_pt = np.inf

        _this_mask = (input_jets_decluster.pt > _min_pt) & (input_jets_decluster.pt < _max_pt)
        pt_masks.append( _this_mask )


    #
    #  Sample the PDFs for the jets we will uncluster
    #
    for _var_name in input_pdfs["varNames"]:

        if _var_name.find("_vs_") == -1:
            is_1d_pdf = True

            _sampled_data = np.ones(n_jets)
            _sampled_data_vs_pT = []
            for _iPt in range(n_pt_bins):
                _sampled_data_vs_pT.append(np.ones(n_jets))
        else:
            is_1d_pdf = False

            _sampled_data_x = np.ones(n_jets)
            _sampled_data_y = np.ones(n_jets)
            _sampled_data_x_vs_pT = []
            _sampled_data_y_vs_pT = []
            for _iPt in range(n_pt_bins):
                _sampled_data_x_vs_pT.append(np.ones(n_jets))
                _sampled_data_y_vs_pT.append(np.ones(n_jets))

        # Sample the pdfs from the different splitting options
        for _splitting_name, _num_samples, _indicies_tuple in splittings:

            for _iPt in range(n_pt_bins):

                if is_1d_pdf:
                    probs   = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["probs"], dtype=float)
                    centers = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["bin_centers"], dtype=float)
                    _sampled_data_vs_pT[_iPt][_indicies_tuple] = np.random.choice(centers, size=_num_samples, p=probs)
                else:
                    probabilities_flat = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["probabilities_flat"], dtype=float)
                    xcenters           = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["xcenters"],      dtype=float)
                    ycenters           = np.array(input_pdfs[_splitting_name][_var_name][_iPt]["ycenters"],      dtype=float)

                    xcenters_flat = np.repeat(xcenters, len(ycenters))
                    ycenters_flat = np.tile(ycenters, len(xcenters))

                    sampled_indices = np.random.choice(len(probabilities_flat), size=_num_samples, p=probabilities_flat)

                    _sampled_data_x_vs_pT[_iPt][_indicies_tuple] = xcenters_flat[sampled_indices]
                    _sampled_data_y_vs_pT[_iPt][_indicies_tuple] = ycenters_flat[sampled_indices]


            #
            #  Now work out which pT bins to use
            #
            if is_1d_pdf:

                for iPt in range(n_pt_bins):
                    _pt_indicies = np.where(ak.flatten(pt_masks[iPt]))[0]
                    _sampled_data[_pt_indicies] = _sampled_data_vs_pT[iPt][_pt_indicies]

            else:

                for iPt in range(n_pt_bins):
                    _pt_indicies = np.where(ak.flatten(pt_masks[iPt]))[0]
                    _sampled_data_x[_pt_indicies] = _sampled_data_x_vs_pT[iPt][_pt_indicies]
                    _sampled_data_y[_pt_indicies] = _sampled_data_y_vs_pT[iPt][_pt_indicies]


        #
        # Save the sampled data to the jets to be uclustered for use in decluster_combined_jets
        #
        if is_1d_pdf:
            input_jets_decluster[_var_name]         = ak.unflatten(_sampled_data,    ak.num(input_jets_decluster))
        else:
            input_jets_decluster["zA"]         = ak.unflatten(_sampled_data_x,    ak.num(input_jets_decluster))
            input_jets_decluster["thetaA"]     = ak.unflatten(_sampled_data_y,    ak.num(input_jets_decluster))
