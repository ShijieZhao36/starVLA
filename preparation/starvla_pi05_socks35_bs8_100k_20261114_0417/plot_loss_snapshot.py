"""Plot actual TensorBoard scalars from the isolated PI05 100k run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    snapshot=json.loads(args.snapshot.read_text())
    rows=snapshot['loss']
    steps=np.array([r['step'] for r in rows],dtype=np.int64)
    loss=np.array([r['value'] for r in rows],dtype=np.float64)
    assert np.all(np.isfinite(loss)) and np.all(loss>=0)
    assert np.all(np.diff(steps)==100) and steps[0]==100
    window=20
    mean=np.convolve(loss,np.ones(window)/window,mode='valid')
    mean_steps=steps[window-1:]
    latest=int(steps[-1])
    recent_start=max(int(steps[0]),latest-20000)
    raw_color='#3978cf'
    mean_color='#d35d23'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#344054','xtick.color':'#475467','ytick.color':'#475467',
        'axes.edgecolor':'#cbd5e1','figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff'})
    fig,axes=plt.subplots(1,2,figsize=(14.2,5.4))
    fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.74,wspace=.23)
    fig.suptitle('StarVLA PI05 - 100k training run',x=.075,y=.965,ha='left',
                 fontsize=19,fontweight='bold',color='#172b4d')
    timestamp=snapshot['snapshot_at'][:19].replace('T',' ')
    fig.text(.075,.89,f'35 socks datasets  |  8 GPUs x batch 8 = global batch 64  |  {timestamp} (UTC+8)',
             color='#475467',fontsize=11)
    fig.text(.075,.835,f'Latest logged: {latest:,} steps   |   Raw loss: {loss[-1]:.5f}   |   Last 20-point mean: {mean[-1]:.5f}',
             color='#172b4d',fontsize=11)
    fmt=FuncFormatter(lambda x,_: '0' if x==0 else f'{x/1000:g}k')
    for ax,title in zip(axes,['Full history','Recent 20k steps']):
        ax.plot(steps,loss,color=raw_color,alpha=.30,lw=1,label='Raw logged loss')
        ax.plot(mean_steps,mean,color=mean_color,lw=2.3,label='Trailing 20-point mean')
        ax.set_title(title,loc='left',pad=12,color='#172b4d',fontweight='bold')
        ax.set_xlabel('Optimizer step',labelpad=9)
        ax.grid(axis='y',color='#e8edf4',lw=.8)
        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.tick_params(axis='both',length=3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Global mean flow-matching loss (14 action dims)',labelpad=11)
    axes[0].set_xlim(0,latest)
    axes[0].set_ylim(0,max(loss)*1.12)
    axes[0].axvline(1000,color='#94a3b8',ls='--',lw=1)
    axes[0].text(1600,max(loss)*1.04,'warmup ends',color='#64748b',fontsize=9,va='top')
    axes[0].legend(frameon=False,loc='upper right',fontsize=10)
    axes[1].set_xlim(recent_start,latest)
    recent=loss[steps>=recent_start]
    lo=max(0,float(recent.min())-.002)
    hi=float(recent.max())+.003
    axes[1].set_ylim(lo,hi)
    axes[1].scatter([latest],[loss[-1]],color=raw_color,s=20,zorder=5)
    axes[1].legend(frameon=False,loc='upper right',fontsize=10)
    fig.text(.075,.055,'TensorBoard: loss/action_fm  |  One logged point per 100 steps  |  Training loss, not held-out validation',
             fontsize=10,color='#667085')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=180)
    plt.close(fig)
    summary={'snapshot_at':snapshot['snapshot_at'],'run_id':snapshot['run_id'],
             'points':len(rows),'last_step':latest,'last_raw_loss':float(loss[-1]),
             'trailing_mean_points':window,'last_trailing_mean':float(mean[-1]),
             'first_20_logged_mean':float(loss[:20].mean()),
             'last_20_logged_mean':float(loss[-20:].mean()),
             'source_tag':snapshot['tag'],'source':snapshot['source']}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
