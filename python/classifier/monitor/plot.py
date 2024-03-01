def plotClasses(train, valid, name, contr=None, selection=''):
    # Make place holder datasets to add the training/validation set graphical distinction to the legend
    trainLegend=pltHelper.dataSet(name=  'Training', color='black', alpha=1.0, linewidth=1)
    validLegend=pltHelper.dataSet(name='Validation', color='black', alpha=0.5, linewidth=2)
    contrLegend=pltHelper.dataSet(name='Control Region', color='black', alpha=0.5, linewidth=1, fmt='o') if contr is not None else None

    extraClasses = []
    binWidth = 0.05
    if classifier in ["SvB",'SvB_MA']:
        extraClasses = [sg,bg]
        bins = np.arange(-binWidth, 1+2*binWidth, binWidth)
    else:
        bins = np.arange(-2*binWidth, 1+2*binWidth, binWidth)

    for cl1 in classes+extraClasses: # loop over classes
        cl1cl2_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': 'P( '+cl1.name+r' $\rightarrow$ Class )',
                       'ylabel': 'Arb. Units',
                       }
        cl2cl1_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': r'P( Class $\rightarrow$ '+cl1.name+' )',
                       'ylabel': 'Arb. Units',
                       }
        for cl2 in classes+extraClasses: # loop over classes
        # Make datasets to be plotted
            cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl1cl2_args['dataSets'] += [cl1cl2_valid, cl1cl2_train]

            cl2cl1_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(train,'w'+cl2.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl2cl1_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(valid,'w'+cl2.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl2cl1_args['dataSets'] += [cl2cl1_valid, cl2cl1_train]

        if classifier in ['FvT']:
            # multijet probabilities well defined but no multijet class labels. Therefore cl1cl2 plot can include multijet but not cl2cl1 plot.
            m4 = classInfo(abbreviation='m4', name= 'FourTag Multijet', color='blue')
            m3 = classInfo(abbreviation='m3', name='ThreeTag Multijet', color='violet')
            for cl2 in [m4,m3]:
                cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
                cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
                cl1cl2_args['dataSets'] += [cl1cl2_train, cl1cl2_valid]

        #make the plotter
        cl1cl2 = pltHelper.histPlotter(**cl1cl2_args)
        cl2cl1 = pltHelper.histPlotter(**cl2cl1_args)
        #remove the lines from the trainLegend/validLegend placeholders
        cl1cl2.artists[0].remove()
        cl1cl2.artists[1].remove()
        cl2cl1.artists[0].remove()
        cl2cl1.artists[1].remove()

        #save the pdf
        try:
            cl1cl2.savefig(name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))
        except:
            print("cannot save", name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))

        try:
            cl2cl1.savefig(name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))


    if classifier in ['FvT']:
        #bins = np.arange(-0.5,5,0.1)
        bins = np.quantile(train.rd4[train.Rd4!=3], np.arange(0,1.05,0.05), interpolation='linear')
        bm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                         'bins': bins,
                         'divideByBinWidth': True,
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        d4_train = pltHelper.dataSet(name=d4.name, points=train.rd4[train.Rd4!=3], weights= train.wd4[train.Rd4!=3]/train_fraction, color=d4.color, alpha=1.0, linewidth=1)
        d4_valid = pltHelper.dataSet(              points=valid.rd4[valid.Rd4!=3], weights= valid.wd4[valid.Rd4!=3]/valid_fraction, color=d4.color, alpha=0.5, linewidth=2)
        # if contr is not None:
        #     bm_vs_d4_args['dataSets'].append(contrLegend)
        #     d4_contr = pltHelper.dataSet(              points=contr.rd4, weights= contr.wd4/contr.w_sum, color=d4.color, alpha=0.5, linewidth=1, fmt='o')
        bm_train = pltHelper.dataSet(name='Background Model', 
                                     points =np.concatenate((train.rd3[train.Rd3!=3], train.rt3[train.Rt3!=3], train.rt4[train.Rt4!=3]),axis=None), 
                                     weights=np.concatenate((train.wd3[train.Rd3!=3],-train.wt3[train.Rt3!=3], train.wt4[train.Rt4!=3]),axis=None)/train_fraction, 
                                     color='brown', alpha=1.0, linewidth=1)
        bm_valid = pltHelper.dataSet(points =np.concatenate((valid.rd3[valid.Rd3!=3], valid.rt3[valid.Rt3!=3], valid.rt4[valid.Rt4!=3]),axis=None), 
                                     weights=np.concatenate((valid.wd3[valid.Rd3!=3],-valid.wt3[valid.Rt3!=3], valid.wt4[valid.Rt4!=3]),axis=None)/valid_fraction, 
                                     color='brown', alpha=0.5, linewidth=2)
        # if contr is not None:
        #     bm_contr = pltHelper.dataSet(points=np.concatenate((contr.rd3,contr.rt3,contr.rt4),axis=None), 
        #                                  weights=np.concatenate((contr.wd3,-contr.wt3,contr.wt4)/contr.w_sum,axis=None), 
        #                                  color='brown', alpha=0.5, linewidth=1, fmt='o')
        t4_train = pltHelper.dataSet(name=t4.name, points=train.rt4[train.Rt4!=3], weights= train.wt4[train.Rt4!=3]/train_fraction, color=t4.color, alpha=1.0, linewidth=1)
        t4_valid = pltHelper.dataSet(              points=valid.rt4[valid.Rt4!=3], weights= valid.wt4[valid.Rt4!=3]/valid_fraction, color=t4.color, alpha=0.5, linewidth=2)
        # if contr is not None:
        #     t4_contr = pltHelper.dataSet(              points=contr.rt4, weights= contr.wt4/contr.w_sum, color=t4.color, alpha=0.5, linewidth=1, fmt='o')
        t3_train = pltHelper.dataSet(name=t3.name, points=train.rt3[train.Rt3!=3], weights=-train.wt3[train.Rt3!=3]/train_fraction, color=t3.color, alpha=1.0, linewidth=1)
        t3_valid = pltHelper.dataSet(              points=valid.rt3[valid.Rt3!=3], weights=-valid.wt3[valid.Rt3!=3]/valid_fraction, color=t3.color, alpha=0.5, linewidth=2)
        # if contr is not None:
        #     t3_contr = pltHelper.dataSet(              points=contr.rt3, weights=-contr.wt3/contr.w_sum, color=t3.color, alpha=0.5, linewidth=1, fmt='o')
        #     bm_vs_d4_args['dataSets'] += [d4_contr, d4_valid, d4_train, 
        #                                   bm_contr, bm_valid, bm_train, 
        #                                   t4_contr, t4_valid, t4_train, 
        #                                   t3_contr, t3_valid, t3_train]
        # else:
        bm_vs_d4_args['dataSets'] += [d4_valid, d4_train, 
                                      bm_valid, bm_train, 
                                      t4_valid, t4_train, 
                                      t3_valid, t3_train]

        bm_vs_d4 = pltHelper.histPlotter(**bm_vs_d4_args)
        bm_vs_d4.artists[0].remove()
        bm_vs_d4.artists[1].remove()
        if contr is not None:
            bm_vs_d4.artists[2].remove()
        try:
            bm_vs_d4.savefig(name.replace('.pdf','_bm_vs_d4.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_bm_vs_d4.pdf'))

        rbm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                         'bins': bins,
                         'divideByBinWidth': True,
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        rbm_train = pltHelper.dataSet(name='Background Model', 
                                     points= np.concatenate((train.rd3[train.Rd3!=3]                        , train.rt4[train.Rt4!=3]),axis=None), 
                                     weights=np.concatenate((train.rd3[train.Rd3!=3]*train.wd3[train.Rd3!=3], train.wt4[train.Rt4!=3]),axis=None)/train_fraction, 
                                     color='brown', alpha=1.0, linewidth=1)
        rbm_valid = pltHelper.dataSet(points=np.concatenate((valid.rd3[valid.Rd3!=3]                        , valid.rt4[valid.Rt4!=3]),axis=None), 
                                     weights=np.concatenate((valid.rd3[valid.Rd3!=3]*valid.wd3[valid.Rd3!=3], valid.wt4[valid.Rt4!=3]),axis=None)/valid_fraction, 
                                     color='brown', alpha=0.5, linewidth=2)
        # if contr is not None:
        #     rbm_vs_d4_args['dataSets'].append(contrLegend)
        #     rbm_contr = pltHelper.dataSet(points=np.concatenate((contr.rd3,contr.rt4),axis=None), 
        #                                   weights=np.concatenate((contr.rd3*contr.wd3,contr.wt4)/contr.w_sum,axis=None), 
        #                                   color='brown', alpha=0.5, linewidth=1, fmt='o')
        rt3_train = pltHelper.dataSet(name=t3.name, points=train.rt3[train.Rt3!=3], weights=-train.rt3[train.Rt3!=3]*train.wt3[train.Rt3!=3]/train_fraction, color=t3.color, alpha=1.0, linewidth=1)
        rt3_valid = pltHelper.dataSet(              points=valid.rt3[valid.Rt3!=3], weights=-valid.rt3[valid.Rt3!=3]*valid.wt3[valid.Rt3!=3]/valid_fraction, color=t3.color, alpha=0.5, linewidth=2)
        # if contr is not None:
        #     rt3_contr = pltHelper.dataSet(              points=contr.rt3, weights=-contr.rt3*contr.wt3/contr.w_sum, color=t3.color, alpha=0.5, linewidth=1, fmt='o')
        #     rbm_vs_d4_args['dataSets'] += [ d4_contr,  d4_valid,  d4_train, 
        #                                     rbm_contr, rbm_valid, rbm_train, 
        #                                     t4_contr,  t4_valid,  t4_train,
        #                                     rt3_contr, rt3_valid, rt3_train]
        # else:
        rbm_vs_d4_args['dataSets'] += [d4_valid,  d4_train, 
                                       rbm_valid, rbm_train, 
                                       t4_valid,  t4_train,
                                       rt3_valid, rt3_train]
        rbm_vs_d4 = pltHelper.histPlotter(**rbm_vs_d4_args)
        rbm_vs_d4.artists[0].remove()
        rbm_vs_d4.artists[1].remove()
        if contr is not None:
            rbm_vs_d4.artists[2].remove()
        try:
            rbm_vs_d4.savefig(name.replace('.pdf','_rbm_vs_d4.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_rbm_vs_d4.pdf'))

        rbm_vs_d4_args['ratio'] = [[2,4],[3,5]]
        rbm_vs_d4_args['ratioRange'] = [0.9,1.1]
        rbm_vs_d4_args['ratioTitle'] = 'd4/bm'
        rbm_vs_d4_args['bins'] = np.arange(0,5.1,0.1)
        rbm_vs_d4_args['overflow'] = True
        rbm_vs_d4_args['divideByBinWidth'] = False
        rbm_vs_d4 = pltHelper.histPlotter(**rbm_vs_d4_args)
        rbm_vs_d4.artists[0].remove()
        rbm_vs_d4.artists[1].remove()
        if contr is not None:
            rbm_vs_d4.artists[2].remove()

        c_valid = pltHelper.histChisquare(obs=d4_valid.points, obs_w=d4_valid.weights,
                                          exp=rbm_valid.points, exp_w=rbm_valid.weights,
                                          bins=rbm_vs_d4_args['bins'], overflow=True)
        c_train = pltHelper.histChisquare(obs=d4_train.points, obs_w=d4_train.weights,
                                          exp=rbm_train.points, exp_w=rbm_train.weights,
                                          bins=rbm_vs_d4_args['bins'], overflow=True)
        rbm_vs_d4.sub1.annotate('$\chi^2/$NDF (Training)[Validation] = (%1.2f, %1.0f$\%%$)[%1.2f, %1.0f$\%%$]'%(c_train.chi2/c_train.ndfs, c_train.prob*100, c_valid.chi2/c_valid.ndfs, c_valid.prob*100), 
                                (1.0,1.02), horizontalalignment='right', xycoords='axes fraction')
        try:
            rbm_vs_d4.savefig(name.replace('.pdf','_rbm_vs_d4_fixedBins.pdf'))
        except:
            print("cannot save",name.replace('.pdf','_rbm_vs_d4_fixedBins.pdf'))



#Simple ROC Curve plot function
def plotROC(train, valid, control=None, plotName='test.pdf'): #fpr = false positive rate, tpr = true positive rate
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    ax.set_title(train.title)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)
    plt.xlabel('Rate( '+valid.trueName+' to '+valid.trueName+' )')
    plt.ylabel('Rate( '+valid.falseName+' to '+valid.falseName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training (%0.4f)"%train.auc)
    ax.plot(valid.tpr, 1-valid.fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation (%0.4f)"%valid.auc)
    if control is not None:
        ax.plot(control.tpr, 1-control.fpr, color='#d34031', linestyle='--', linewidth=2, alpha=0.5, label="Control (%0.4f)"%control.auc)
    ax.legend(loc='lower left')
    #ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(valid.auc))

    if valid.sigma is not None:
        #ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        #ax.text(rate_StoS+0.03, rate_BtoB-0.100, ZB+"SR \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
        ax.scatter(valid.tprSigma, (1-valid.fprSigma), marker='o', c='#d34031')
        ax.text(valid.tprSigma+0.03, (1-valid.fprSigma)-0.025, 
                ("(%0.3f, %0.3f), "+valid.pName+" $>$ %0.2f \n S=%0.1f, B=%0.1f, $%1.2f\sigma$")%(valid.tprSigma, (1-valid.fprSigma), valid.thrSigma, valid.S, valid.B, valid.sigma), 
                bbox=bbox)

    try:
        f.savefig(plotName)
    except:
        print("Cannot save fig: ",plotName)

    plt.close(f)

def makePlots(self, baseName='', suffix=''):
        self.modelPkl = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_epoch%02d.pkl'%(self.name, self.epoch)
        if not baseName: baseName = self.modelPkl.replace('.pkl', '')
        if classifier in ['SvB','SvB_MA']:
            plotROC(self.training.roc1,    self.validation.roc1,    plotName=baseName+suffix+'_ROC_sb.pdf')
            plotROC(self.training.roc2,    self.validation.roc2,    plotName=baseName+suffix+'_ROC_zz_zh.pdf')
            plotROC(self.training.roc_zz,  self.validation.roc_zz,  plotName=baseName+suffix+'_ROC_zz.pdf')
            plotROC(self.training.roc_zh,  self.validation.roc_zh,  plotName=baseName+suffix+'_ROC_zh.pdf')
            plotROC(self.training.roc_hh,  self.validation.roc_hh,  plotName=baseName+suffix+'_ROC_hh.pdf')
            plotROC(self.training.roc_SR,  self.validation.roc_SR,  plotName=baseName+suffix+'_ROC_SR.pdf')
        if classifier in ['DvT3']:
            plotROC(self.training.roc_t3, self.validation.roc_t3, plotName=baseName+suffix+'_ROC_t3.pdf')
        if classifier in ['DvT4']:
            plotROC(self.training.roc_t4, self.validation.roc_t4, plotName=baseName+suffix+'_ROC_t4.pdf')
        if classifier in ['FvT']:
            # plotROC(self.training.roc_td, self.validation.roc_td, control=self.control.roc_td, plotName=baseName+suffix+'_ROC_td.pdf')
            # plotROC(self.training.roc_43, self.validation.roc_43, control=self.control.roc_43, plotName=baseName+suffix+'_ROC_43.pdf')
            # plotROC(self.training.roc_d43, self.validation.roc_d43, control=self.control.roc_d43, plotName=baseName+suffix+'_ROC_d43.pdf')
            plotROC(self.training.roc_td, self.validation.roc_td, plotName=baseName+suffix+'_ROC_td.pdf')
            plotROC(self.training.roc_43, self.validation.roc_43, plotName=baseName+suffix+'_ROC_43.pdf')
            plotROC(self.training.roc_d43, self.validation.roc_d43, plotName=baseName+suffix+'_ROC_d43.pdf')
        plotClasses(self.training, self.validation, baseName+suffix+'.pdf', contr=self.control)

        if self.training.cross_entropy is not None:
            plotCrossEntropy(self.training, self.validation, baseName+suffix+'.pdf')


def plotCrossEntropy(train, valid, name):
    cross_entropy_train = pltHelper.dataSet(name=  'Training Set', points=train.cross_entropy*train.w, weights=train.w/train.w_sum, color='black', alpha=1.0, linewidth=1)
    cross_entropy_valid = pltHelper.dataSet(name='Validation Set', points=valid.cross_entropy*valid.w, weights=valid.w/valid.w_sum, color='black', alpha=0.5, linewidth=2)

    w_train_notzero = (train.w!=0)
    bins = np.quantile(train.cross_entropy[w_train_notzero]*train.w[w_train_notzero], np.arange(0,1.05,0.05), interpolation='linear')

    cross_entropy_args = {'dataSets': [cross_entropy_train, cross_entropy_valid],
                          'bins': bins,#[b/50.0 for b in range(0,76)],
                          'xlabel': r'Cross Entropy * Event Weight',
                          'ylabel': 'Arb. Units',
                          'divideByBinWidth': True,
                          }

    for cl1 in classes: # loop over classes
        w_train = getattr(train,'w'+cl1.abbreviation)
        w_valid = getattr(valid,'w'+cl1.abbreviation)
        ce_train = getattr(train,'ce'+cl1.abbreviation)
        ce_valid = getattr(valid,'ce'+cl1.abbreviation)
        cl1_train = pltHelper.dataSet(name=cl1.name, points=ce_train*w_train, weights=w_train/train.w_sum, color=cl1.color, alpha=1.0, linewidth=1)
        cl1_valid = pltHelper.dataSet(               points=ce_valid*w_valid, weights=w_valid/valid.w_sum, color=cl1.color, alpha=0.5, linewidth=2)
        cross_entropy_args['dataSets'] += [cl1_valid, cl1_train]

    cross_entropy = pltHelper.histPlotter(**cross_entropy_args)
    try:
        cross_entropy.savefig(name.replace('.pdf','_cross_entropy.pdf'))
    except:
        print("cannot save",name.replace('.pdf','_cross_entropy.pdf'))


 def plotByEpoch(self, train, valid, ylabel, suffix, loc='best', control=None, batch=None):
        fig = plt.figure(figsize=(10,7))

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        #plt.ylim(yMin,yMax)
        if batch:
            plt.plot([b[0] for b in batch], [b[1] for b in batch], 
                     linestyle='-',
                     linewidth=1, alpha=0.33,
                     color='black',
                     label='Training Batches')

        x = np.arange(1,self.epoch+1)
        plt.plot(x, train,
                 marker="o",
                 linestyle="-",
                 linewidth=1, alpha=1.0,
                 color="#d34031",
                 label="Training Set")
        plt.plot(x, valid,
                 marker="o",
                 linestyle="-",
                 linewidth=2, alpha=0.5,
                 color="#d34031",
                 label="Validation Set")
        if control:
            plt.plot(x, control,
                     marker="o",
                     linestyle="--",
                     linewidth=2, alpha=0.5,
                     color="#d34031",
                     label="Control Region")

        plt.xticks(x)
        #plt.yticks(np.linspace(-1, 1, 5))
        plt.legend(loc=loc)

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        if 'norm' in suffix:
            ylim=[0.8, 1.2]

        for e in self.bs_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        for e in self.lr_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1, zorder=1)
        if 'norm' in suffix:
            plt.plot(xlim, [1,1], color='k', alpha=1.0, linestyle='-', linewidth=0.75, zorder=0)
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        plotName = 'ZZ4b/nTupleAnalysis/pytorchModels/%s_%s.pdf'%(self.name, suffix)
        try:
            fig.savefig(plotName)
        except:
            print("Cannot save fig: ",plotName)
        plt.close(fig)        
