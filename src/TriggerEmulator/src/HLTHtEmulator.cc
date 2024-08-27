#include "HLTHtEmulator.h"

#include <TRandom3.h>
#include <TF1.h>
#include <TSystem.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <assert.h>

using namespace TriggerEmulator;
using std::cout; using std::endl;

HLTHtEmulator::HLTHtEmulator(std::string histName_, std::string fileName_, bool debug){
  histName = histName_;
  fileName = fileName_;

  m_rand = new TRandom3();
  //m_mode = mode;
  m_debug = debug;

  //if(m_mode == 0){
  //  m_htCut = htCut;
  //}else if(m_mode == 1){
  //  m_sigmoid = new TF1(("func"+name).c_str(),"(1.0/(1+ TMath::Exp(-[0]*(x-[1]))))", -301, 14000);
  //  m_sigmoid->SetParameters(p0, p1);
  //}else if(m_mode == 2){


  TFile* inputFile = new TFile(gSystem->ExpandPathName(("$CMSSW_BASE/src/TriggerEmulator/nTupleAnalysis/data/"+fileName).c_str()),"READ");
  if(m_debug) inputFile->ls();

  TGraphAsymmErrors* relativeEff = dynamic_cast<TGraphAsymmErrors*>(inputFile->Get(histName.c_str()));
  if(!relativeEff){
    inputFile->ls();
    cout << "HLTBTagEmulator::failed to get histogram: " << histName << " from  file: " << fileName << endl;
  }

  assert(relativeEff && "Failed to retrieve histogram");
  if(m_debug) std::cout << "relativeEff  is " << relativeEff << std::endl;
    
  if(m_debug){
    std::cout << "Filling the bims" << std::endl;  
    std::cout << relativeEff->GetN() << std::endl;  
  }

  assert(relativeEff->GetN() && "histogram empty");

  unsigned int nBins = relativeEff->GetN();
  for(unsigned int iBin = 0; iBin<nBins; ++iBin){
    double eff, ht;

    relativeEff->GetPoint(iBin, ht, eff);
    //float ht_low   = relativeEff->GetErrorXlow(iBin);
    float ht_high  = relativeEff->GetErrorXhigh(iBin);
    float err_low  = relativeEff->GetErrorYlow(iBin);
    float err_high = relativeEff->GetErrorYhigh(iBin);
    float err_ave  = (err_low+err_high)/2;
    float error_total = err_ave;
    if(m_debug) std::cout << "Getting Point " << iBin << std::endl;  

    //if(sf_tag){
    //  double sf, ptSF;
    //  m_sf_tag->GetPoint(iBin, ptSF, sf);
    //  //float ptSF_low   = m_sf_tag->GetErrorXlow(iBin);
    //  //float ptSF_high  = m_sf_tag->GetErrorXhigh(iBin);
    //  float errSF_low  = m_sf_tag->GetErrorYlow(iBin);
    //  float errSF_high = m_sf_tag->GetErrorYhigh(iBin);
    //  float errSF_ave  = (errSF_low+errSF_high)/2;
    //
    //  eff *= sf;
    //  error_total = sqrt(err_ave*err_ave + errSF_ave*errSF_ave);
    //}

    if(m_debug)
      std::cout << "\tiBin " << iBin << " ht: " << ht-ht_high << " - " << ht << " - " << ht+ht_high
		<< " eff: " << eff << " +/- " << error_total
		<< std::endl;
    m_highBinEdge .push_back(ht+ht_high);
    assert((eff >= 0) && "ERROR Loading eff < 0");
    m_eff        .push_back(eff);
    m_effErr     .push_back(error_total);
  }
  
  if(m_debug) std::cout << "Closing ROOTFile" << std::endl;
  inputFile->Close();
  delete inputFile;

}



bool HLTHtEmulator::passHt(float ht, float seedOffset, float smearFactor){

  //int seed = (int)(pt * seedOffset + pt); 
  //m_rand->SetSeed(seed);
  double htRand = m_rand->Rndm();

  return passHtThreshold(ht, htRand, smearFactor);
}


bool HLTHtEmulator::passHtThreshold(float ht, double htRand, float smearFactor){  
  bool debug = false;

  //
  // => (m_mode == 2)
  //
  //thisTagEff = tagEff->Eval(pt)*scaleFactor;
  float eff    = -99;
  float effErr = -99;

  for(unsigned int iBin = 0; iBin< m_highBinEdge.size(); ++iBin){
    if(debug) cout << iBin <<  "comparing " << ht << " to " <<  m_highBinEdge.at(iBin) << endl;
    if(ht < m_highBinEdge.at(iBin)){
      eff    = m_eff   .at(iBin);
      effErr = m_effErr.at(iBin);
      if(debug) cout << " eff is " << eff <<endl;
      break;
    }
  }
  if(eff < 0) {
    eff    = m_eff.back();
    effErr = m_effErr.back(); 
  }
  assert((eff >= 0) && "ERROR eff < 0");



  float thisTagEff = eff + effErr*smearFactor;
  if(debug) cout << " thisTagEff  " <<  thisTagEff << " for ht = " << ht << endl;
  if(thisTagEff > htRand)
    return true;

  return false;
}
