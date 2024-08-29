#include "TrigEmulator.h"
#include <iostream>

using namespace TriggerEmulator;
using std::cout; using std::endl;
TrigEmulator::TrigEmulator(std::string name, 
			   std::vector<HLTHtEmulator* > htThresholds, 
			   std::vector<HLTJetEmulator* > jetThresholds, std::vector<unsigned int> jetMultiplicities, 
			   std::vector<HLTBTagEmulator* > bTagOpPoints, std::vector<unsigned int> bTagMultiplicities,
			   int nToys){
  m_trigName = name;
  m_htThresholds = htThresholds;
  m_jetThresholds = jetThresholds;
  m_jetMultiplicities = jetMultiplicities;
  m_bTagOpPoints = bTagOpPoints;
  m_bTagMultiplicities = bTagMultiplicities;
  m_nToys = nToys;
}

bool TrigEmulator::passTrig(const std::vector<float>& offline_jet_pts, const std::vector<float>& offline_btagged_jet_pts, const float ht, float seedOffset){
  //std::cout << "TrigEmulator:: " << m_trigName << " ht " << ht << std::endl;

  //
  //  Ht Cut s
  //
  for(unsigned int iThres = 0; iThres < m_htThresholds.size(); ++iThres){
    HLTHtEmulator* HLTHtCut = m_htThresholds.at(iThres);

    if(ht > 0 && HLTHtCut){
      if(!HLTHtCut->passHt(ht, seedOffset)) {
	//std::cout << "Fail ht  " << std::endl;
	return false;
      }
    }
  }


  //
  // Loop on all thresholds
  //
  for(unsigned int iThres = 0; iThres < m_jetThresholds.size(); ++iThres){
    HLTJetEmulator* HLTJet = m_jetThresholds.at(iThres);
          
    unsigned int nJetsPassed = 0;
      
    //
    // Count passing jets
    //
    for(float jet_pt: offline_jet_pts){
      if(HLTJet->passJet(jet_pt,seedOffset)) ++nJetsPassed;
    }
      
    //
    //  Impose trigger cut
    //
    if(nJetsPassed < m_jetMultiplicities.at(iThres)){
      return false;
    }
  }

  //
  //  Apply BTag Operating Points
  //
  for(unsigned int iThres = 0; iThres < m_bTagOpPoints.size(); ++iThres){
    HLTBTagEmulator* HLTBTag = m_bTagOpPoints.at(iThres);

    unsigned int nJetsPassBTag = 0;
      
    //
    // Count passing jets
    //
    for(float bjet_pt: offline_btagged_jet_pts){
      if(HLTBTag->passJet(bjet_pt,seedOffset)) ++nJetsPassBTag;
    }
      
    //
    //  Impose trigger cut
    //
    if(nJetsPassBTag < m_bTagMultiplicities.at(iThres)){
      return false;
    }
  }

  return true;
}

//
// Used for calculating correlated decisions with input (ht and btagging) weights
//
bool TrigEmulator::passTrig(const std::vector<float>& offline_jet_pts, const std::vector<float>& offline_btagged_jet_pts, const float ht, 
			    const std::vector<std::vector<double> >& btag_rand, const std::vector<double>& ht_rand, float seedOffset
			    ){
  //std::cout << "TrigEmulator:: " << m_trigName << " ht " << ht << std::endl;

  //
  //  Ht Cut s
  //
  for(unsigned int iThres = 0; iThres < m_htThresholds.size(); ++iThres){
    HLTHtEmulator* HLTHtCut = m_htThresholds.at(iThres);

    if(ht > 0 && HLTHtCut){
      if(!HLTHtCut->passHtThreshold(ht, ht_rand.at(iThres))) {
	//std::cout << "Fail ht  " << std::endl;
	return false;
      }
    }
  }


  //
  // Loop on all thresholds
  //
  for(unsigned int iThres = 0; iThres < m_jetThresholds.size(); ++iThres){
    HLTJetEmulator* HLTJet = m_jetThresholds.at(iThres);
          
    unsigned int nJetsPassed = 0;
      
    //
    // Count passing jets
    //
    for(float jet_pt: offline_jet_pts){
      if(HLTJet->passJet(jet_pt,seedOffset)) ++nJetsPassed;
    }
      
    //
    //  Impose trigger cut
    //
    if(nJetsPassed < m_jetMultiplicities.at(iThres)){
      return false;
    }
  }


  //
  //  Apply BTag Operating Points
  //
  for(unsigned int iThres = 0; iThres < m_bTagOpPoints.size(); ++iThres){
    HLTBTagEmulator* HLTBTag = m_bTagOpPoints.at(iThres);

    unsigned int nJetsPassBTag = 0;
      
    //
    // Count passing jets
    //
    for(unsigned int iBJet = 0; iBJet < offline_btagged_jet_pts.size(); ++iBJet){
      float bjet_pt = offline_btagged_jet_pts.at(iBJet);
      if(HLTBTag->passJetThreshold(bjet_pt,btag_rand.at(iBJet).at(iThres))) ++nJetsPassBTag;
    }
      
    //
    //  Impose trigger cut
    //
    if(nJetsPassBTag < m_bTagMultiplicities.at(iThres)){
      return false;
    }
  }

  return true;
}


float TrigEmulator::calcWeight(const std::vector<float>& offline_jet_pts, const std::vector<float>& offline_btagged_jet_pts, float ht){

  unsigned int nPass  = 0;
  
  for(unsigned int iToy = 0; iToy < m_nToys; ++iToy){

    // 
    // Count all events
    //
    if(passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy))
      ++nPass;
  }
  //cout << "TrigEmulator::calcWeight is " << float(nPass)/m_nToys << endl;
  return float(nPass)/m_nToys;
}



void TrigEmulator::Fill(const std::vector<float>& offline_jet_pts, const std::vector<float>& offline_btagged_jet_pts, float ht){

  for(unsigned int iToy = 0; iToy < m_nToys; ++iToy){
    
    // 
    // Count all events
    //
    ++m_nTotal;
    if(passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, iToy))
      ++m_nPass;
  }

  return;
}

void TrigEmulator::dumpResults(){
  std::cout << "eff_rel = " << float(m_nPass)/m_nTotal << "    (" << m_nPass << ")"<< std::endl;
}

