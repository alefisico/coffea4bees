#include "TrigEmulatorTool.h"
#include "HLTJetEmulator.h"
#include <iostream>
#include <assert.h>
#include <array>
#include <TRandom3.h>

using namespace TriggerEmulator;

using std::cout;  using std::endl;
using std::vector;  using std::string; using std::array;

TrigEmulatorTool::TrigEmulatorTool(string name, unsigned int nToys = 1000, string year = "2018", bool debug=false, bool useMCTurnOns=false) :  m_name(name),  m_nToys(nToys), m_debug(debug), m_useMCTurnOns(useMCTurnOns)
{
  m_rand = new TRandom3();

  if(year == "2018"){
    config2018();
  }else if(year == "2017"){
    config2017();
  }else if(year == "2016"){
    config2016();
  }else{
    cout << "TrigEmulatorTool::ERROR year has to be 2018, 2017 or 2016. Not " << year << endl;
  }

  //
  //  Emulated Jet Thresholds Triggers
  // 
  if(debug) cout << "TrigEmulatorTool::Making Jet Thresholds " << endl;
  for (auto const& jc : JetConfig){
    m_Jet.insert(std::make_pair(jc.first,new HLTJetEmulator(jc.second.at(0),jc.second.at(1),debug)));
  }

  //
  //  Emulated HT Thresholds
  //
  if(debug) cout << "TrigEmulatorTool::Making Ht Thresholds " << endl;
  for (auto const& hc : HTConfig){
    m_Ht.insert(std::make_pair(hc.first,new HLTHtEmulator(hc.second.at(0),hc.second.at(1),debug)));
  }

  //
  //  Emulated BTag Thresholds
  //
  if(debug) cout << "TrigEmulatorTool::Making BTag Thresholds " << endl;
  for (auto const& bc : BTagConfig){
    m_BTag.insert(std::make_pair(bc.first,new HLTBTagEmulator(bc.second.at(0),bc.second.at(1),debug)));
  }


  if(debug) cout << "TrigEmulatorTool::Leaving " << endl;
}


// Just HT 
void TrigEmulatorTool::AddTrig(string trigName,  const vector<hTTurnOn>& HTNames){
			       
  if(m_debug) cout << "TrigEmulatorTool::AddTrig Ht Only " << trigName << endl;
  return AddTrig(trigName, HTNames, vector<jetTurnOn>(), vector<unsigned int>(), vector<bTagTurnOn>(), vector<unsigned int>());
}


// jets
void TrigEmulatorTool::AddTrig(string trigName,  
			       const vector<jetTurnOn>&  JetNames, const vector<unsigned int>& JetMults){
  if(m_debug) cout << "TrigEmulatorTool::AddTrig Jets Only " << trigName << endl;
  return AddTrig(trigName, vector<hTTurnOn>(), JetNames, JetMults, vector<bTagTurnOn>(), vector<unsigned int>());
}

// HT and Jets
void TrigEmulatorTool::AddTrig(string trigName,  const vector<hTTurnOn>& HTNames, 
			       const vector<jetTurnOn>&  JetNames, const vector<unsigned int>& JetMults
			       ){
  if(m_debug) cout << "TrigEmulatorTool::AddTrig Ht and Jets " << trigName << endl;
  return AddTrig(trigName, HTNames, JetNames, JetMults, vector<bTagTurnOn>(), vector<unsigned int>());
}

// jets and btags
void TrigEmulatorTool::AddTrig(string trigName,  
			       const vector<jetTurnOn>&  JetNames, const vector<unsigned int>& JetMults, 
			       const vector<bTagTurnOn>&  TagNames, const vector<unsigned int>& TagMults){
  if(m_debug) cout << "TrigEmulatorTool::AddTrig Jet and btags " << trigName << endl;
  return AddTrig(trigName, vector<hTTurnOn>(), JetNames, JetMults, TagNames, TagMults);
}



void TrigEmulatorTool::AddTrig(string trigName,  const vector<hTTurnOn>& HTNames, 
			       const vector<jetTurnOn>&  JetNames, const vector<unsigned int>& JetMults,
			       const vector<bTagTurnOn>&  TagNames, const vector<unsigned int>& TagMults
			       ){
  if(m_debug) cout << "TrigEmulatorTool::AddTrig Enter " << trigName << endl;
  if(m_emulatedTrigMenu.find(trigName) != m_emulatedTrigMenu.end()){
    cout << "TrigEmulatorTool::AddTrig ERROR " << trigName << " already included " << endl;
    return;
  }
  if(m_debug) cout << "TrigEmulatorTool::AddTrig ADding trig " << trigName << endl;
  assert(JetNames.size() == JetMults.size());
  assert(TagNames.size() == TagMults.size());
  assert(TagNames.size() < 3 );
    

  //
  // Config The Ht
  //
  vector<HLTHtEmulator*> HTCuts; 
  for(const hTTurnOn& HTName : HTNames){
    if(m_Ht.find(HTName) == m_Ht.end()){
      //static_cast<std::underlying_type<A>::type>(a)
      cout << "TrigEmulatorTool::AddTrig ERROR Ht " << static_cast<std::underlying_type<hTTurnOn>::type>(HTName) << " not defined " << endl;
      assert(false);
    }  
    HTCuts.push_back(m_Ht[HTName]);
  }

  //
  // Config The Jets
  //
  if(m_debug) cout << "TrigEmulatorTool::AddTrig making vetor of  HLTJetEmulators " << endl;
  vector<HLTJetEmulator*> JetPtCuts; 
  for(const jetTurnOn& jetName : JetNames){
    if(m_Jet.find(jetName) == m_Jet.end()){
      cout << "TrigEmulatorTool::AddTrig ERROR Jet " << static_cast<std::underlying_type<jetTurnOn>::type>(jetName) << " not defined " << endl;
      assert(false);
    }  
    JetPtCuts.push_back(m_Jet[jetName]);
  }

  // 
  // Config the BTags
  //
  vector<HLTBTagEmulator*> BTagPtCuts; 
  for(const bTagTurnOn& btagName : TagNames){
    if(m_BTag.find(btagName) == m_BTag.end()){
      cout << "TrigEmulatorTool::AddTrig ERROR BTag " << static_cast<std::underlying_type<bTagTurnOn>::type>(btagName) << " not defined " << endl;
      assert(false);
    }  
    BTagPtCuts.push_back(m_BTag[btagName]);
  }

  if(m_debug) cout << "TrigEmulatorTool::AddTrig inserting  " << endl;
  m_emulatedTrigMenu .insert(std::make_pair(trigName, new TrigEmulator(trigName, HTCuts, JetPtCuts,  JetMults, BTagPtCuts, TagMults, m_nToys)));
  m_emulatedDecisions.insert(std::make_pair(trigName, false));
  m_emulatedWeights  .insert(std::make_pair(trigName, 0.0));
  
}

void TrigEmulatorTool::SetDecisions(const  vector<float>& offline_jet_pts, const vector<float>& offline_btagged_jet_pts, float ht){
  for (auto trigIt=m_emulatedTrigMenu.begin(); trigIt!=m_emulatedTrigMenu.end(); ++trigIt){
    if(trigIt->second->passTrig(offline_jet_pts, offline_btagged_jet_pts, ht)){
      m_emulatedDecisions[trigIt->first] = true;
    }else{
      m_emulatedDecisions[trigIt->first] = false;
    }
    
  }
  return;
}



bool TrigEmulatorTool::GetDecision(string trigName){
  if(m_emulatedDecisions.find(trigName) == m_emulatedDecisions.end()){
    cout << "TrigEmulatorTool::GetDecisions ERROR " << trigName << " not defined. Returning false " << endl;
    return false;
  }
  return m_emulatedDecisions[trigName];
}


void TrigEmulatorTool::SetWeights(const vector<float>& offline_jet_pts, const vector<float>& offline_btagged_jet_pts, float ht){
  for (auto trigIt=m_emulatedTrigMenu.begin(); trigIt!=m_emulatedTrigMenu.end(); ++trigIt){
    m_emulatedWeights[trigIt->first] = trigIt->second->calcWeight(offline_jet_pts, offline_btagged_jet_pts, ht);
  }
  return;
}


float TrigEmulatorTool::GetWeight(string trigName){
  if(m_emulatedWeights.find(trigName) == m_emulatedWeights.end()){
    cout << "TrigEmulatorTool::GetWeight ERROR " << trigName << " not defined. Returning 0.0 " << endl;
    return 0.0;
  }
  return m_emulatedWeights[trigName];
}

float TrigEmulatorTool::GetWeightOR(const vector<float>& offline_jet_pts, const vector<float>& offline_btagged_jet_pts, float ht, bool setSeed){
  
  unsigned int nPass  = 0;

  for(unsigned int iToy = 0; iToy < m_nToys; ++iToy){
    
    bool passAny = false;

    //
    // jet pt weights can be uncorrelated among triggers
    //   

    // 
    //  b-tag weights should be correlated (separate numbers for Calo and PF btagging)
    // 
    vector<vector<double> > btag_weights = getRandWeights(offline_btagged_jet_pts, setSeed, 2*iToy);

    //
    //  The L1 ht should be correlated among triggers 
    //
    if(setSeed){
      int seed = (int)(ht * (3*iToy) + ht);
      m_rand->SetSeed(seed);
    }
    // L1 HT / Calo Ht / PF Ht
    vector<double> ht_weights = {m_rand->Rndm(), m_rand->Rndm(), m_rand->Rndm()};


    for (auto trigIt=m_emulatedTrigMenu.begin(); trigIt!=m_emulatedTrigMenu.end(); ++trigIt){

      // 
      // Count all events
      //
      if(trigIt->second->passTrig(offline_jet_pts, offline_btagged_jet_pts, ht, btag_weights, ht_weights,  iToy))
	passAny = true;
	
    }

    if(passAny)
      ++nPass;
  }
  
  return float(nPass)/m_nToys;
}



void TrigEmulatorTool::Fill(const vector<float>& offline_jet_pts, const vector<float>& offline_btagged_jet_pts, float ht){
  for (auto trigIt=m_emulatedTrigMenu.begin(); trigIt!=m_emulatedTrigMenu.end(); ++trigIt)
    trigIt->second->Fill(offline_jet_pts, offline_btagged_jet_pts, ht);
  return;
}


void TrigEmulatorTool::dumpResults(){
  for (auto trigIt=m_emulatedTrigMenu.begin(); trigIt!=m_emulatedTrigMenu.end(); ++trigIt){
    std::cout << trigIt->first << " ";
    trigIt->second->dumpResults();
  }

  return;
}





void TrigEmulatorTool::config2018(){

  cout << "TrigEmulatorTool::configuring for 2018 " << endl;

  std::string fileName2018 = m_useMCTurnOns ? "haddOutput_All_MC2018_11Nov_fittedTurnOns.root" : "haddOutput_All_Data2018_11Nov_fittedTurnOns.root" ;
  cout << "TrigEmulatorTool::using file \t " << fileName2018 << endl;

  //
  //  Jet Emulator
  //
  JetConfig = {
    {jetTurnOn::PF30BTag,                   {{"pt_s_PF30inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF40BTag,                   {{"pt_s_PF40inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF45BTag,                   {{"pt_s_PF45inMJMatchBtagTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF60BTag,                   {{"pt_PF60inMJMatchBtagTandP_jetID_TurnOn",                       fileName2018}} },
    {jetTurnOn::PF75BTag,                   {{"pt_PF75inMJMatchBtagTandP_jetID_TurnOn",                       fileName2018}} },
    {jetTurnOn::PF116BTag,                  {{"pt_PF116inMJMatchBtagTandP_jetID_TurnOn",                      fileName2018}} },
    {jetTurnOn::PF116DrBTag,                {{"pt_PF116DrfilterMatchBtagTandP_jetID_TurnOn",                    fileName2018}} },

    {jetTurnOn::L1112BTag,   {{"pt_L12b112inMJMatchBtagTandP_jetID_TurnOn", fileName2018}} },
    {jetTurnOn::Calo100BTag, {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn", fileName2018}} },


    {jetTurnOn::PF30,                   {{"pt_s_PF30inMJTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF40,                   {{"pt_s_PF40inMJTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF45,                   {{"pt_s_PF45inMJTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF60,                   {{"pt_PF60inMJTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF75,                   {{"pt_PF75inMJTandP_jetID_TurnOn",                     fileName2018}} },
    {jetTurnOn::PF116,                  {{"pt_PF116inMJTandP_jetID_TurnOn",                     fileName2018}} },

    {jetTurnOn::L1112,   {{"pt_L12b112inMJTandP_jetID_TurnOn", fileName2018}} },
    {jetTurnOn::Calo100, {{"pt_Calo100inMJTandP_jetID_TurnOn", fileName2018}} },


  };

  //
  //  BTag Emulator
  //
  BTagConfig = {
    {bTagTurnOn::PFDeepCSV,                 {{"pt_PFDeepCSVinMJMatchBtagTandP_jetID_TurnOn",               fileName2018}} },
    {bTagTurnOn::CaloDeepCSV,               {{"pt_CaloDeepCSVinMJMatchBtagTandP_jetID_TurnOn",             fileName2018}} },
    {bTagTurnOn::CaloDeepCSV2b116,          {{"pt_CaloDeepCSVinMJ2b116MatchBtagTandP_jetID_TurnOn",        fileName2018}} },
    //{bTagTurnOn::Calo100BTag,               {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn",                 fileName2018}} },
    {bTagTurnOn::Calo100BTag,           {{"pt_Calo100ANDCaloCSVDeepinMJMatchBtagTandP_jetID_TurnOn",             fileName2018}} },

    {bTagTurnOn::PFDeepCSVloose,             {{"pt_PFDeepCSVinMJMatchBlooseTandP_jetID_TurnOn",               fileName2018}} },
    {bTagTurnOn::CaloDeepCSVloose,           {{"pt_CaloDeepCSVinMJMatchBlooseTandP_jetID_TurnOn",             fileName2018}} },
    {bTagTurnOn::CaloDeepCSV2b116loose,      {{"pt_CaloDeepCSVinMJ2b116MatchBlooseTandP_jetID_TurnOn",        fileName2018}} },
    {bTagTurnOn::Calo100BTagloose,           {{"pt_Calo100ANDCaloCSVDeepinMJMatchBlooseTandP_jetID_TurnOn",   fileName2018}} },

  };

  //
  // Ht Config
  //
  HTConfig = {
    //{hTTurnOn::L1ORAll_Ht330_4j_3b,            {{"hT30_L1ORAll_TurnOn",      fileName2018}} },
    {hTTurnOn::L1ORAll_Ht330_4j_3b,       {{"hT30_L1ORAll_TurnOn_4Jet2Tag", fileName2018}} },
    //{hTTurnOn::L1ORAll_Ht330_4j_3b,            {{"hT30_L1ORAll_TurnOn_2Tag",      fileName2018}} },
    {hTTurnOn::CaloHt320,                      {{"hT30_CaloHt320_TurnOn",    fileName2018}} },
    {hTTurnOn::PFHt330,                        {{"hT30_PFHt330_TurnOn",      fileName2018}} },
  };


}


void TrigEmulatorTool::config2017(){

  cout << "TrigEmulatorTool::configuring for 2017 " << endl;
  std::string fileName2017 = m_useMCTurnOns ? "haddOutput_All_MC2017_11Nov_fittedTurnOns.root" : "haddOutput_All_Data2017_11Nov_fittedTurnOns.root" ;
  cout << "TrigEmulatorTool::using file \t " << fileName2017 << endl;

  //
  //  Jet Emulator
  //
  JetConfig = {
    {jetTurnOn::PF30BTag,                   {{"pt_s_PF30inMJMatchBtagTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF40BTag,                   {{"pt_s_PF40inMJMatchBtagTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF45BTag,                   {{"pt_s_PF45inMJMatchBtagTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF60BTag,                   {{"pt_PF60inMJMatchBtagTandP_jetID_TurnOn",                      fileName2017}} },
    {jetTurnOn::PF75BTag,                   {{"pt_PF75inMJMatchBtagTandP_jetID_TurnOn",                      fileName2017}} },
    {jetTurnOn::PF100BTag,                  {{"pt_PF100inMJMatchBtagTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF100DrBTag,                {{"pt_PF100DrfilterMatchBtagTandP_jetID_TurnOn",                  fileName2017}} },

    {jetTurnOn::L1100BTag,   {{"pt_L12b100inMJTandP_jetID_TurnOn", fileName2017}} },
    {jetTurnOn::Calo100BTag, {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn", fileName2017}} },
 
    {jetTurnOn::PF30,                   {{"pt_s_PF30inMJTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF40,                   {{"pt_s_PF40inMJTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF45,                   {{"pt_s_PF45inMJTandP_jetID_TurnOn",                     fileName2017}} },
    {jetTurnOn::PF60,                   {{"pt_PF60inMJTandP_jetID_TurnOn",                       fileName2017}} },
    {jetTurnOn::PF75,                   {{"pt_PF75inMJTandP_jetID_TurnOn",                       fileName2017}} },
  };


  //
  //  BTag Emulator
  //
  BTagConfig = {
    {bTagTurnOn::PFCSV,                 {{"pt_PFCSVinMJMatchBtagTandP_jetID_TurnOn",               fileName2017}} },
    {bTagTurnOn::CaloCSV,               {{"pt_CaloCSVinMJMatchBtagTandP_jetID_TurnOn",             fileName2017}} },
    {bTagTurnOn::CaloCSV2b100,          {{"pt_CaloCSVinMJ2b100MatchBtagTandP_jetID_TurnOn",        fileName2017}} },
    {bTagTurnOn::Calo100BTag,           {{"pt_Calo100ANDCaloCSVinMJMatchBtagTandP_jetID_TurnOn",             fileName2017}} },


    {bTagTurnOn::PFCSVloose,                 {{"pt_PFCSVinMJMatchBlooseTandP_jetID_TurnOn",               fileName2017}} },
    {bTagTurnOn::CaloCSVloose,               {{"pt_CaloCSVinMJMatchBlooseTandP_jetID_TurnOn",             fileName2017}} },
    {bTagTurnOn::CaloCSV2b100loose,          {{"pt_CaloCSVinMJ2b100MatchBlooseTandP_jetID_TurnOn",        fileName2017}} },
    {bTagTurnOn::Calo100BTagloose,           {{"pt_Calo100ANDCaloCSVinMJMatchBlooseTandP_jetID_TurnOn",             fileName2017}} },

  };

  //
  // Ht Config
  //
  HTConfig = {
    //{hTTurnOn::L1ORAll_Ht300_4j_3b,       {{"hT30_L1ORAll_TurnOn",      fileName2017}} },
    {hTTurnOn::L1ORAll_Ht300_4j_3b,       {{"hT30_L1ORAll_TurnOn_4Jet2Tag", fileName2017}} },
    {hTTurnOn::CaloHt300,                 {{"hT30_CaloHt300_TurnOn",    fileName2017}} },
    {hTTurnOn::PFHt300,                   {{"hT30_PFHt300_TurnOn",      fileName2017}} },
  };

}



void TrigEmulatorTool::config2016(){

  cout << "TrigEmulatorTool::configuring for 2016 " << endl;
  std::string fileName2016 = m_useMCTurnOns ? "haddOutput_All_MC2016_11Nov_fittedTurnOns.root" : "haddOutput_All_Data2016_11Nov_fittedTurnOns.root" ;
  cout << "TrigEmulatorTool::using file \t " << fileName2016 << endl;

  //
  //  Jet Emulator
  //
  JetConfig = {
    {jetTurnOn::Calo30BTag, {{"pt_Calo30inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },
    {jetTurnOn::Calo45BTag, {{"pt_Calo45inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },
    {jetTurnOn::Calo90BTag, {{"pt_Calo90inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },
    {jetTurnOn::Calo100BTag, {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },

    {jetTurnOn::PF30BTag,   {{"pt_PF30inMJMatchBtagTandP_jetID_TurnOn",                     fileName2016}} },
    {jetTurnOn::PF45BTag,   {{"pt_PF45inMJMatchBtagTandP_jetID_TurnOn",                     fileName2016}} },
    {jetTurnOn::PF90BTag,   {{"pt_PF90inMJMatchBtagTandP_jetID_TurnOn",                       fileName2016}} },
    {jetTurnOn::PF100BTag,  {{"pt_PF100inMJMatchBtagTandP_jetID_TurnOn",                       fileName2016}} },

    {jetTurnOn::L1100BTag,   {{"pt_L12b100inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },
  };

  //
  //  BTag Emulator
  //
  BTagConfig = {
    {bTagTurnOn::CaloCSV,               {{"pt_CaloCSVinMJMatchBtagTandP_jetID_TurnOn",             fileName2016}} },
    {bTagTurnOn::CaloCSV2b100,          {{"pt_CaloCSVinMJ2b100MatchBtagTandP_jetID_TurnOn",        fileName2016}} },
    {bTagTurnOn::Calo100BTag,           {{"pt_Calo100inMJMatchBtagTandP_jetID_TurnOn", fileName2016}} },

    {bTagTurnOn::CaloCSVloose,               {{"pt_CaloCSVinMJMatchBlooseTandP_jetID_TurnOn",             fileName2016}} },
    {bTagTurnOn::CaloCSV2b100loose,          {{"pt_CaloCSVinMJ2b100MatchBlooseTandP_jetID_TurnOn",        fileName2016}} },
    //{bTagTurnOn::Calo100BTagloose,           {{"pt_Calo100inMJMatchBlooseTandP_jetID_TurnOn", fileName2016}} },
  };

  //
  // HT Config
  //
  HTConfig = {
    {hTTurnOn::L1ORAll_4j_3b,       {{"hT30_L1ORAll_TurnOn_4Jet2Tag", fileName2016}} },
    //{hTTurnOn::L1ORAll_4j_3b,       {{"hT30_L1ORAll_TurnOn",      fileName2016}} },
    {hTTurnOn::L1ORAll_2j_2j_3b,    {{"hT30_L1ORAll_2j_2j_3b_TurnOn_4Jet2Tag",      fileName2016}} },
    //{hTTurnOn::L1ORAll_2j_2j_3b,    {{"hT30_L1ORAll_2j_2j_3b_TurnOn",      fileName2016}} },
  };



}

std::vector<std::vector<double> >  TrigEmulatorTool::getRandWeights(const std::vector<float>& input, bool setSeed, int seedOffset){
  if(setSeed && input.size()){
    int seed = (int)(input.at(0) * seedOffset + input.at(0));
    m_rand->SetSeed(seed);
  }
  
  vector<vector<double> > randNumbers;
  for(unsigned int i=0; i < input.size(); ++i){
    randNumbers.push_back(vector<double>());

    randNumbers.back().push_back(m_rand->Rndm());// Calo 
    randNumbers.back().push_back(m_rand->Rndm());// PF
  }

  return  randNumbers;
}


