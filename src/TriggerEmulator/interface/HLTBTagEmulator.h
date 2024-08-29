// -*- C++ -*-
#if !defined(HLTBTagEmulator_H)
#define HLTBTagEmulator_H

#include <string>
#include <vector>

class TRandom3;

namespace TriggerEmulator {

  class HLTBTagEmulator {
      
  public:

    HLTBTagEmulator(std::string histName_, std::string fileName_, bool debug=false);

    bool passJet(float pt, float seedOffset = 1.0, float smearFactor = 0.0);
    bool passJetThreshold(float pt, double bTagRand, float smearFactor = 0.0);

    std::string histName;
    std::string fileName;

  private:

    TRandom3*           m_rand        = nullptr;

    // EFficeincies and Uncertianties
    std::vector<double> m_highBinEdge;
    std::vector<double> m_eff;
    std::vector<double> m_effErr;

  };

}

#endif // HLTBTagEmulator_H
