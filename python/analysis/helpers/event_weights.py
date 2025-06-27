from coffea.analysis_tools import Weights
from analysis.helpers.common import apply_btag_sf
import correctionlib
import awkward as ak
import numpy as np
import uproot
import logging

def add_weights(event, do_MC_weights: bool = True,
                dataset: str = None,
                year_label: str = None,
                estart: int = 0,
                estop: int = None,
                corrections_metadata: dict = None,
                apply_trigWeight: bool = True,
                friend_trigWeight: callable = None,
                isTTForMixed: bool = False,
                target: callable = None,
                ):
    """Add weights to the event.
    """

    weights = Weights(len(event), storeIndividual=True)
    list_weight_names = []

    if do_MC_weights:
        # genWeight
        lumi    = event.metadata.get('lumi',    1.0)
        xs      = event.metadata.get('xs',      1.0)
        kFactor = event.metadata.get('kFactor', 1.0)
        weights.add( "genweight", event.genWeight * (lumi * xs * kFactor / event.metadata["genEventSumw"]) )
        list_weight_names.append('genweight')
        logging.debug( f"genweight {weights.partial_weight(include=['genweight'])[:10]}\n" )
        logging.debug( f" = {event.genWeight} * ({lumi} * {xs} * {kFactor} / {event.metadata['genEventSumw']})\n")

        # trigger Weight (to be updated)
        if apply_trigWeight:
            trigWeight = event.trigWeight if "trigWeight" in event.fields else friend_trigWeight.arrays(target) if friend_trigWeight else logging.error(f"No friend tree for trigWeight found.")

            hlt = ak.where(event.passHLT, 1., 0.)
            weights.add( "CMS_bbbb_resolved_ggf_triggerEffSF",
                        trigWeight.Data, ##* ak.where(trigWeight.MC != 0, hlt / trigWeight.MC, 1) ### uncomment for new data.
                        trigWeight.MC,
                        hlt )
            list_weight_names.append('CMS_bbbb_resolved_ggf_triggerEffSF')
            logging.debug( f"trigWeight {weights.partial_weight(include=['CMS_bbbb_resolved_ggf_triggerEffSF'])[:10]}\n" )

        # puWeight (to be checked)
        if not isTTForMixed:
            puWeight = list( correctionlib.CorrectionSet.from_file( corrections_metadata["PU"] ).values() )[0]
            weights.add( f"CMS_pileup_{year_label}",
                            puWeight.evaluate(event.Pileup.nTrueInt, "nominal"),
                            puWeight.evaluate(event.Pileup.nTrueInt, "up"),
                            puWeight.evaluate(event.Pileup.nTrueInt, "down"), )
            list_weight_names.append(f"CMS_pileup_{year_label}")
            logging.debug( f"PU weight {weights.partial_weight(include=[f'CMS_pileup_{year_label}'])[:10]}\n" )

        # L1 prefiring weight
        if ( "L1PreFiringWeight" in event.fields ):  #### AGE: this should be temprorary (field exists in UL)
            weights.add( f"CMS_prefire_{year_label}",
                            event.L1PreFiringWeight.Nom,
                            event.L1PreFiringWeight.Up,
                            event.L1PreFiringWeight.Dn, )
            logging.debug( f"L1Prefire weight {weights.partial_weight(include=[f'CMS_prefire_{year_label}'])[:10]}\n" )
            list_weight_names.append(f"CMS_prefire_{year_label}")

        if ( "PSWeight" in event.fields ):  #### AGE: this should be temprorary (field exists in UL)
            nom      = np.ones(len(weights.weight()))
            up_isr   = np.ones(len(weights.weight()))
            down_isr = np.ones(len(weights.weight()))
            up_fsr   = np.ones(len(weights.weight()))
            down_fsr = np.ones(len(weights.weight()))

            if len(event.PSWeight[0]) == 4:
                up_isr   = event.PSWeight[:, 0]
                down_isr = event.PSWeight[:, 2]
                up_fsr   = event.PSWeight[:, 1]
                down_fsr = event.PSWeight[:, 3]

            else:
                logging.warning( f"PS weight vector has length {len(event.PSWeight[0])}" )

            weights.add("ps_isr", nom, up_isr, down_isr)
            weights.add("ps_fsr", nom, up_fsr, down_fsr)
            list_weight_names.append(f"ps_isr")
            list_weight_names.append(f"ps_fsr")

        # pdf_Higgs_ggHH, alpha_s, PDFaS weights are included in datacards through the inference tool. Kept this code for reference.
        # if "LHEPdfWeight" in event.fields:

        #     # https://github.com/nsmith-/boostedhiggs/blob/a33dca8464018936fbe27e86d52c700115343542/boostedhiggs/corrections.py#L53
        #     nom  = np.ones(len(weights.weight()))
        #     up   = np.ones(len(weights.weight()))
        #     down = np.ones(len(weights.weight()))

        #     # NNPDF31_nnlo_hessian_pdfas
        #     # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
        #     if "306000 - 306102" in event.LHEPdfWeight.__doc__:
        #         # Hessian PDF weights
        #         # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
        #         arg = event.LHEPdfWeight[:, 1:-2] - np.ones( (len(weights.weight()), 100) )

        #         summed = ak.sum(np.square(arg), axis=1)
        #         pdf_unc = np.sqrt((1.0 / 99.0) * summed)
        #         weights.add("pdf_Higgs_ggHH", nom, pdf_unc + nom)

        #         # alpha_S weights
        #         # Eq. 27 of same ref
        #         as_unc = 0.5 * ( event.LHEPdfWeight[:, 102] - event.LHEPdfWeight[:, 101] )

        #         weights.add("alpha_s", nom, as_unc + nom)

        #         # PDF + alpha_S weights
        #         # Eq. 28 of same ref
        #         pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
        #         weights.add("PDFaS", nom, pdfas_unc + nom)

        #     else:
        #         weights.add("alpha_s", nom, up, down)
        #         weights.add("pdf_Higgs_ggHH", nom, up, down)
        #         weights.add("PDFaS", nom, up, down)
        #     list_weight_names.append(f"alpha_s")
        #     list_weight_names.append(f"pdf_Higgs_ggHH")
        #     list_weight_names.append(f"PDFaS")
    else:
        weights.add("data", np.ones(len(event)))
        list_weight_names.append(f"data")

    logging.debug(f"weights event {weights.weight()[:10]}")
    logging.debug(f"Weight Statistics {weights.weightStatistics}")

    return weights, list_weight_names


def add_pseudotagweights(
    event, 
    weights, 
    JCM: callable = None,
    JCM_lowpt: callable = None, 
    apply_FvT: bool = False, 
    isDataForMixed: bool = False, 
    list_weight_names: list = [], 
    event_metadata: dict = {},
    year_label: str = None, 
    len_event: int = None, 
    label3b: str = "threeTag"
):
    """
    Add pseudo-tagging weights to the selected events and update the weights object.

    Args:
        event (awkward.Array): Selected events.
        weights (Weights): Weights object to store the calculated weights.
        JCM (callable, optional): Jet Combinatoric Model function for pseudo-tagging. Defaults to None.
        apply_FvT (bool, optional): Whether to apply FvT weights. Defaults to False.
        JCM_lowpt (callable, optional): Jet Combinatoric Model function for low-pt jets. Defaults to None.
        isDataForMixed (bool, optional): Whether the data is for mixed events. Defaults to False.
        list_weight_names (list, optional): List to store the names of added weights. Defaults to an empty list.
        event_metadata (dict, optional): Metadata for the event. Defaults to an empty dictionary.
        year_label (str, optional): Year label for the dataset. Defaults to None.
        len_event (int, optional): Total number of events. Defaults to None.
        label3b (str, optional): Label for three-tag events. Defaults to "threeTag".

    Returns:
        tuple: Updated weights object and list of weight names.
    """

    # Calculate weight without JCM and FvT
    all_weights = ['genweight', 'CMS_bbbb_resolved_ggf_triggerEffSF', f'CMS_pileup_{year_label}', 'CMS_btag']
    logging.debug( f"noJCM_noFVT partial {weights.partial_weight(include=all_weights)[:10]}" )
    event["weight_noJCM_noFvT"] = weights.partial_weight(include=all_weights)

    if JCM:
        # Calculate pseudo-tagging weights
        event["Jet_untagged_loose"] = event.Jet[event.Jet.selected & ~event.Jet.tagged_loose]
        pseudoTagWeight = np.full(len(event), event.weight)  # Initialize with existing weights
        nJet_pseudotagged = np.zeros(len(event), dtype=int)

        pseudoTagWeight[event[label3b]], nJet_pseudotagged[event[label3b]] = JCM(
            event[event[label3b]]['Jet_untagged_loose'], 
            event.event[event[label3b]]
        )
        event["nJet_pseudotagged"] = nJet_pseudotagged
        event["pseudoTagWeight"] = pseudoTagWeight
        logging.debug( f"pseudoTagWeight {event.pseudoTagWeight[:10]}\n" )
        logging.debug( f"nJet_pseudotagged {event.nJet_pseudotagged[:10]}\n" )

        # Update number of tagged jets
        nTagJets = ak.num(event.tagJet, axis=1).to_numpy()
        nTagJets[event[label3b]] = ak.num(event.tagJet_loose[event[label3b]], axis=1)
        event["nJet_ps_and_tag"] = nJet_pseudotagged + nTagJets

        event["pseudoTagWeight_lowpt"] = np.ones(len(event), dtype=float)  # Initialize dummy lowpt pseudoTagWeight

        if 'lowpt' in label3b:
            event["weight_noJCM_lowpt_noFvT"] = event.weight_noJCM_noFvT * event.pseudoTagWeight

            if JCM_lowpt:
                event["Jet_untagged_loose_lowpt"] = event.Jet[event.Jet.selected_lowpt & ~event.Jet.tagged_loose_lowpt]
                pseudoTagWeight_lowpt = np.full(len(event), event.weight)  # Initialize with existing weights
                nJet_pseudotagged_lowpt = np.zeros(len(event), dtype=int)

                pseudoTagWeight_lowpt[event[label3b]], nJet_pseudotagged_lowpt[event[label3b]] = JCM_lowpt(
                    event[event[label3b]]['Jet_untagged_loose_lowpt'], 
                    event.event[event[label3b]]
                )
                event["nJet_pseudotagged_lowpt"] = nJet_pseudotagged_lowpt
                event["pseudoTagWeight_lowpt"] = pseudoTagWeight_lowpt
                logging.debug( f"pseudoTagWeight_lowpt {event.pseudoTagWeight_lowpt[:10]}\n" )
                logging.debug( f"nJet_pseudotagged_lowpt {event.nJet_pseudotagged_lowpt[:10]}\n" )

                # Update number of tagged jets
                nTagJets_lowpt = ak.num(event.tagJet_lowpt, axis=1).to_numpy()
                nTagJets_lowpt[event[label3b]] = ak.num(event.tagJet_loose_lowpt[event[label3b]], axis=1)
                event["nJet_ps_and_tag_lowpt"] = nJet_pseudotagged_lowpt + nTagJets_lowpt

        # Calculate weight without FvT
        weight_noFvT = ak.where(event[label3b], event.weight * event.pseudoTagWeight * event.pseudoTagWeight_lowpt, event.weight)

        event["weight_noFvT"] = weight_noFvT
        logging.debug( f"weight_noFvT {event.weight_noFvT[:10]}\n" )

        # Apply FvT weights if required
        if apply_FvT:
            if isDataForMixed:
                for _JCM_load, _FvT_name in zip(event_metadata["JCM_loads"], event_metadata["FvT_names"]):
                    event[f"weight_{_FvT_name}"] = np.where(
                        event[label3b],
                        event.weight * getattr(event, f"{_JCM_load}") * getattr(getattr(event, _FvT_name), _FvT_name),
                        event.weight,
                    )

                weight_JCM = np.where(
                    event[label3b], 
                    getattr(event, f"{event_metadata['JCM_loads'][0]}"), 
                    1.0
                )
                weights.add("JCM", weight_JCM)
                list_weight_names.append("JCM")
                logging.debug( f"JCM {weights.partial_weight(include=['JCM'])[:10]}\n" )

                weight_FvT = np.where(
                    event[label3b], 
                    event.FvT.FvT, 
                    1.0
                )
                weights.add("FvT", weight_FvT)
                list_weight_names.append("FvT")
                logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )

            else:
                weight = np.where(
                    event[label3b], 
                    event["pseudoTagWeight"] * event["pseudoTagWeight_lowpt"] * event.FvT.FvT, 
                    1.0
                )
                weights.add("FvT", weight)
                list_weight_names.append("FvT")
                logging.debug( f"FvT {weights.partial_weight(include=['FvT'])[:10]}\n" )
        else:
            weight_noFvT = np.copy(event.weight)
            weight_noFvT = np.where(
                event[label3b], 
                event["pseudoTagWeight"] * event["pseudoTagWeight_lowpt"], 
                1.0
            )
            weights.add("no_FvT", weight_noFvT)
            list_weight_names.append("no_FvT")
            logging.debug( f"no_FvT {weights.partial_weight(include=['no_FvT'])[:10]}\n" )

    return weights, list_weight_names

def add_btagweights( event, weights,
                    list_weight_names: list = [],
                    shift_name: str = None,
                    run_systematics: bool = False,
                    use_prestored_btag_SF: bool = False,
                    corrections_metadata: dict = None,
                    ):

    if use_prestored_btag_SF:
        weights.add( "CMS_btag", event.CMSbtag )
    else:

        sys_value = "central"
        if shift_name and ( 'CMS_scale_j_' in shift_name ):
            if 'Down' in shift_name:
                sys_value = f"down_jes{shift_name.replace('CMS_scale_j_', '').replace('Down', '')}"
            elif 'Up' in shift_name:
                sys_value = f"up_jes{shift_name.replace('CMS_scale_j_', '').replace('Up', '')}"
        logging.debug(f"shift_name: {shift_name}, sys_value: {sys_value}\n\n")

        btag_SF_weights = apply_btag_sf(
            event.selJet_no_bRegCorr,
            sys_value="central",
            correction_file=corrections_metadata["btagSF"],
            btag_uncertainties=corrections_metadata["btag_uncertainties"] if (not shift_name) & run_systematics else None
        )

        if (not shift_name) & run_systematics:
            weights.add_multivariation( f"CMS_btag", btag_SF_weights["btagSF_central"],
                                        corrections_metadata["btag_uncertainties"],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_up" in name ],
                                        [ var.to_numpy() for name, var in btag_SF_weights.items() if "_down" in name ], )
        else:
            weights.add( "CMS_btag", btag_SF_weights["btagSF_central"] )

    list_weight_names.append(f"CMS_btag")

    return weights, list_weight_names
