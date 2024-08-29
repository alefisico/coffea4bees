#include "HLTBTagEmulator.h"

#include <TGraphAsymmErrors.h>
#include <TRandom3.h>
#include <TFile.h>
#include <TSystem.h>
#include <iostream>
#include <assert.h>

using std::cout; using std::endl;
using namespace TriggerEmulator;

HLTBTagEmulator::HLTBTagEmulator(std::string histName_, std::string fileName_, bool debug){
  if(debug) std::cout << "HLTBTagEmulator:: from File: " << fileName_ << " and histogram " << histName_ << std::endl;
  histName = histName_;
  fileName = fileName_;
  m_rand = new TRandom3();

  if(fileName=="none"){
    m_highBinEdge.push_back(-1.0);
    m_eff        .push_back( 1.0);
    m_effErr     .push_back( 0.0);
    return;
  }

  TFile* inputFile = new TFile(gSystem->ExpandPathName(("$CMSSW_BASE/src/TriggerEmulator/nTupleAnalysis/data/"+fileName).c_str()),"READ");
  if(debug) inputFile->ls();

  TGraphAsymmErrors* relativeEff = dynamic_cast<TGraphAsymmErrors*>(inputFile->Get(histName.c_str()));
  if(!relativeEff){
    inputFile->ls();
    cout << "HLTBTagEmulator::failed to get histogram: " << histName << " from  file: " << fileName << endl;
  }


  assert(relativeEff && "Failed to retrieve histogram");
  if(debug) std::cout << "relativeEff  is " << relativeEff << std::endl;

  if(debug){
    std::cout << "Filling the bims" << std::endl;  
    std::cout << relativeEff->GetN() << std::endl;  
  }

  assert(relativeEff->GetN() && "histogram empty");

  unsigned int nBins = relativeEff->GetN();
  for(unsigned int iBin = 0; iBin<nBins; ++iBin){
    double eff, pt;

    relativeEff->GetPoint(iBin, pt, eff);
    //float pt_low   = relativeEff->GetErrorXlow(iBin);
    float pt_high  = relativeEff->GetErrorXhigh(iBin);
    float err_low  = relativeEff->GetErrorYlow(iBin);
    float err_high = relativeEff->GetErrorYhigh(iBin);
    float err_ave  = (err_low+err_high)/2;
    float error_total = err_ave;
    if(debug) std::cout << "Getting Point " << iBin << std::endl;  

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

    if(debug)
      cout << "\tiBin " << iBin << " pt: " << pt-pt_high << " - " << pt << " - " << pt+pt_high
	     << " eff: " << eff << " +/- " << error_total
	     << endl;
    m_highBinEdge .push_back(pt+pt_high);
    m_eff        .push_back(eff);
    m_effErr     .push_back(error_total);
  }
  

  if(debug) std::cout << "Closing ROOTFile" << std::endl;
  inputFile->Close();
  delete inputFile;
  if(debug) std::cout << "Left HLTBTagEmulator" << std::endl;
}

bool HLTBTagEmulator::passJet(float pt, float seedOffset, float smearFactor){

  //int seed = (int)(pt * seedOffset + pt); 
  //m_rand->SetSeed(seed);
  double bTagRand = m_rand->Rndm();
  return passJetThreshold(pt, bTagRand, smearFactor);
}


bool HLTBTagEmulator::passJetThreshold(float pt, double bTagRand, float smearFactor){

  //thisTagEff = tagEff->Eval(pt)*scaleFactor;
  float eff    = -99;
  //float sf     = -99;
  float effErr = -99;
  //cout << " pt is " << pt << endl;
  for(unsigned int iBin = 0; iBin< m_highBinEdge.size(); ++iBin){
    if(pt < m_highBinEdge.at(iBin)){
      eff    = m_eff   .at(iBin);
      effErr = m_effErr.at(iBin);
      break;
    }
  }
  if(eff < 0) {
    eff    = m_eff.back();
    effErr = m_effErr.back();
  }
  if(eff < 0) eff = 0;
  //assert((eff > 0) && "ERROR eff < 0");
  //cout << " pt_eff is " << eff << endl;
  

  float thisTagEff = eff + effErr*smearFactor;
  if(thisTagEff > bTagRand)
    return true;

  return false;
}


