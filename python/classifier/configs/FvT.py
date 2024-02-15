from ..dataset import Label, LabelCollection

__all__ = ['Labels']

# die loss
# fC_SB = torch.FloatTensor([wd4_SB/w_SB, wd3_SB/w_SB, wt4_SB/w_SB, wt3_SB/w_SB])
# fC_SR = torch.FloatTensor([             wd3_SR/w_SR, wt4_SR/w_SR, wt3_SR/w_SR])
# loaded_die_loss_SB = -(fC_SB*fC_SB.log()).sum()
# print("fC_SB:",fC_SB)
# print('loaded die loss outside SR:',loaded_die_loss_SB)
# loaded_die_loss_SR = -(fC_SR*fC_SR.log()).sum()
# print("fC_SR:",fC_SR)
# log.print('loaded die loss inside SR: %f'%loaded_die_loss_SR)
# loaded_die_loss = (loaded_die_loss_SB * w_SB + loaded_die_loss_SR * w_SR)/w


class Labels(LabelCollection):
    d4 = Label('FourTag Data')
    d3 = Label('ThreeTag Data')
    t4 = Label(R'FourTag $t\bar{t}$')
    t3 = Label(R'ThreeTag $t\bar{t}$')
