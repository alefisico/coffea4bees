import json
import argparse
import logging
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

# Load CMS style including color-scheme (it's an editable dict)
plt.style.use(hep.style.CMS)

if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser( description='Convert json hist to root TH1F',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output",
                        default="limit_plot.png", help='Output file and directory.')
    parser.add_argument('-i', '--input_files', dest='input_files', nargs='+',
                        default=[], help="Json files with limits")
    parser.add_argument('-l', '--labels', dest='labels', nargs='+',
                        default=[], help="Labels for each json file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")


    fig, ax = plt.subplots()

    ticks = []
    ticks_labels = []
    for i, ifile in enumerate(args.input_files):
        limit = json.load(open(ifile, 'rb'))['120.0']
        label= args.labels[i] if args.labels[i] else ifile

        plt.vlines( 1, i, i+1, color='tab:gray', linestyle='dashed' )
        #plt.plot( [i+.5], [limit[obs]], color='k', marker='o', zorder=10, label=('observed' if i==0 else '')  )
        ticks.append( i+.5 )
        ticks_labels.append(label)
        plt.vlines( limit['exp0'], i, i+0.98, color='k', linestyle='dashed', label=('expected' if i==0 else '') )
        plt.text( limit['exp0']+1.5, i+0.5, np.around(limit['exp0'], decimals=2), ha = 'center')
        plt.fill_betweenx( [ i, i+0.98], 2*[limit['exp+2']], 2*[limit['exp-2']], color = '#FFDF7Fff', label=('68% expected' if i==0 else '' ) )
        plt.fill_betweenx( [ i, i+0.98], 2*[limit['exp+1']], 2*[limit['exp-1']], color = '#85D1FBff', label=('95% expected' if i==0 else '' )  )

    ax.set_yticks( ticks )
    ax.set_yticklabels( ticks_labels )
    ax.set_xlim([0, 15])
    ax.set_ylim([0, len(ticks)+1])
    hep.cms.label("Preliminary", data = True)
    fig.tight_layout()

    # Style
    plt.legend()
    plt.savefig(args.output)
