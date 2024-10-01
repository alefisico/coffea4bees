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
            if ("GluGlu" in dataset) and ("trigWeight" not in event.fields):
                if friend_trigWeight:
                    trigWeight = friend_trigWeight.arrays(target)
                    weights.add( 'CMS_bbbb_resolved_ggf_triggerEffSF',
                                trigWeight.Data,
                                trigWeight.MC,
                                ak.where(event.passHLT, 1., 0.) )
                else:
                    logging.error(f"No friend tree for trigWeight found.")

            else:
                weights.add( "CMS_bbbb_resolved_ggf_triggerEffSF",
                            event.trigWeight.Data,
                            event.trigWeight.MC,
                            ak.where(event.passHLT, 1.0, 0.0), )
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

        if "LHEPdfWeight" in event.fields:

            # https://github.com/nsmith-/boostedhiggs/blob/a33dca8464018936fbe27e86d52c700115343542/boostedhiggs/corrections.py#L53
            nom  = np.ones(len(weights.weight()))
            up   = np.ones(len(weights.weight()))
            down = np.ones(len(weights.weight()))

            # NNPDF31_nnlo_hessian_pdfas
            # https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_hessian_pdfas/NNPDF31_nnlo_hessian_pdfas.info
            if "306000 - 306102" in event.LHEPdfWeight.__doc__:
                # Hessian PDF weights
                # Eq. 21 of https://arxiv.org/pdf/1510.03865v1.pdf
                arg = event.LHEPdfWeight[:, 1:-2] - np.ones( (len(weights.weight()), 100) )

                summed = ak.sum(np.square(arg), axis=1)
                pdf_unc = np.sqrt((1.0 / 99.0) * summed)
                weights.add("pdf_Higgs_ggHH", nom, pdf_unc + nom)

                # alpha_S weights
                # Eq. 27 of same ref
                as_unc = 0.5 * ( event.LHEPdfWeight[:, 102] - event.LHEPdfWeight[:, 101] )

                weights.add("alpha_s", nom, as_unc + nom)

                # PDF + alpha_S weights
                # Eq. 28 of same ref
                pdfas_unc = np.sqrt(np.square(pdf_unc) + np.square(as_unc))
                weights.add("PDFaS", nom, pdfas_unc + nom)

            else:
                weights.add("alpha_s", nom, up, down)
                weights.add("pdf_Higgs_ggHH", nom, up, down)
                weights.add("PDFaS", nom, up, down)
            list_weight_names.append(f"alpha_s")
            list_weight_names.append(f"pdf_Higgs_ggHH")
            list_weight_names.append(f"PDFaS")
    else:
        weights.add("data", np.ones(len(event)))
        list_weight_names.append(f"data")

    logging.debug(f"weights event {weights.weight()[:10]}")
    logging.debug(f"Weight Statistics {weights.weightStatistics}")

    return weights, list_weight_names


def add_pseudotagweights( selev, weights,
                         analysis_selections,
                         JCM: callable = None,
                         apply_FvT: bool = False,
                         isDataForMixed: bool = False,
                         list_weight_names:list = [],
                         event_metadata:dict = {},
                         year_label: str = None,
                         len_event: int = None,
):

    #
    # calculate pseudoTagWeight for threeTag events
    #
    all_weights = ['genweight', 'CMS_bbbb_resolved_ggf_triggerEffSF', f'CMS_pileup_{year_label}' ,'CMS_btag']
    logging.debug( f"noJCM_noFVT partial {weights.partial_weight(include=all_weights)[ analysis_selections ][:10]}" )
    selev["weight_noJCM_noFvT"] = weights.partial_weight( include=all_weights )[analysis_selections]

    if JCM:
        selev["Jet_untagged_loose"] = selev.Jet[ selev.Jet.selected & ~selev.Jet.tagged_loose ]
        nJet_pseudotagged = np.zeros(len(selev), dtype=int)
        pseudoTagWeight = np.ones(len(selev))
        pseudoTagWeight[selev.threeTag], nJet_pseudotagged[selev.threeTag] = ( JCM( selev[selev.threeTag]["Jet_untagged_loose"], selev.event[selev.threeTag], ) )
        selev["nJet_pseudotagged"] = nJet_pseudotagged
        selev["pseudoTagWeight"] = pseudoTagWeight

        weight_noFvT = np.array(selev.weight.to_numpy(), dtype=float)
        weight_noFvT[selev.threeTag] = ( selev.weight[selev.threeTag] * selev.pseudoTagWeight[selev.threeTag] )
        selev["weight_noFvT"] = weight_noFvT

        # Apply pseudoTagWeight and FvT efficiently
        if apply_FvT:
            if isDataForMixed:
                for _JCM_load, _FvT_name in zip(event_metadata["JCM_loads"], event_metadata["FvT_names"]):
                    selev[f"weight_{_FvT_name}"] = np.where(
                        selev.threeTag,
                        selev.weight * getattr(selev, f"{_JCM_load}") * getattr(getattr(selev, _FvT_name), _FvT_name),
                        selev.weight,
                    )
                selev["weight"] = selev[f'weight_{event_metadata["FvT_names"][0]}']
            else:
                weight = np.full(len_event, 1.0)
                weight[analysis_selections] = np.where(selev.threeTag, selev["pseudoTagWeight"] * selev.FvT.FvT, 1.0)
                weights.add("FvT", weight)
                list_weight_names.append("FvT")
        else:
            weight_noFvT = np.full(len_event, 1.0)
            # JA do we really want this line v
            #weight_noFvT[analysis_selections] = np.where(selev.threeTag, selev.weight * selev["pseudoTagWeight"], selev.weight)
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

        btag_SF_weights = apply_btag_sf(
            event.selJet_no_bRegCorr,
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
