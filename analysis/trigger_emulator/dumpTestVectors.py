
def dumpTriggerTestVectors(selev):
    nEvent = 15
    testVectorEvents = selev[selev.threeTag]
    print(f'{chunk}\n\n')
    print(f'{chunk} self.sel_jet_pt      = { [testVectorEvents.selJet.pt[iE].tolist() for iE in range(nEvent)] }')
    print(f'{chunk} self.can_jet_pt      = { [testVectorEvents.canJet.pt[iE].tolist() for iE in range(nEvent)] }')
    print(f'{chunk} self.hT              = { [testVectorEvents.hT[iE]                 for iE in range(nEvent)] }')
    print(f'{chunk} self.hT_trigger      = { [testVectorEvents.hT_trigger[iE]         for iE in range(nEvent)] }')
    print(f'{chunk} self.trigWeightMC    = { [testVectorEvents.trigWeight.MC[iE]      for iE in range(nEvent)] }')
    print(f'{chunk} self.trigWeightData  = { [testVectorEvents.trigWeight.Data[iE]    for iE in range(nEvent)] }')
    print(f'{chunk}\n\n')
