// -*- C++ -*-
#if !defined(HLTHtEmulator_H)
#define HLTHtEmulator_H

#include <string>
#include <vector>

class TRandom3;
class TF1;

namespace TriggerEmulator {

  class HLTHtEmulator {
      
  public:

    HLTHtEmulator(std::string histName_, std::string fileName_, bool debug=false );

    bool passHt(float ht, float seedOffset=1.0, float smearFactor = 0.0);
    bool passHtThreshold(float ht, double htRand, float smearFactor = 0.0);

    std::string histName;
    std::string fileName;
    bool m_debug;

  private:
    TRandom3*   m_rand = nullptr;
    //TF1*        m_sigmoid = nullptr;
    //int         m_mode            = 0; // 0 = Flat Pt Cut  /  1 = Sigmoid / 2 = TGraph
    //float       m_htCut = -1;

    // EFficeincies and Uncertianties
    //std::vector<double> m_lowBinEdge;
    std::vector<double> m_highBinEdge;
    std::vector<double> m_eff;
    std::vector<double> m_effErr;

  };

}

#endif // HLTHtEmulator_H
