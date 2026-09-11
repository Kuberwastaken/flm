"""Plot measured body sensor positions and engineered stimulus checks; no learning scores."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.audit_food_sensor_probe import audit
from flm.food_sensors import FoodField,Source

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'reports/food-sensors/visible-odor-probe.json'; report=json.loads(source.read_text(encoding='utf8')); audit(report)
    sources=[Source(**{key:tuple(value) if isinstance(value,list) else value for key,value in row.items()}) for row in report['field']['sources']]
    field=FoodField(sources); first=report['frames'][0]; antennae=np.asarray(first['antennae_mm'])
    x=np.linspace(-1,12,220); y=np.linspace(-6,6,220); xx,yy=np.meshgrid(x,y)
    height=float(antennae[:,2].mean())
    raw=field.odor_at(np.column_stack((xx.ravel(),yy.ravel(),np.full(xx.size,height))))
    values=(raw/(1+raw)).reshape(*xx.shape,2)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'axes.labelcolor':'#303531','text.color':'#303531','axes.edgecolor':'#aaaaa1','savefig.facecolor':'#faf9f4',
        'svg.hashsalt':'flm-food-sensors-v1'})
    fig=plt.figure(figsize=(15,5.6),facecolor='#faf9f4')
    grid=fig.add_gridspec(1,3,width_ratios=[1,1,1.65],left=.055,right=.97,top=.79,bottom=.22,wspace=.32)
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    for channel,ax in enumerate(axes[:2]):
        contour=ax.contourf(xx,yy,values[:,:,channel],levels=np.linspace(0,.5,11),cmap='Blues' if channel==0 else 'YlOrBr')
        for i,item in enumerate(sources):
            ax.plot(*item.position_mm[:2],'o',color='#303531',ms=4)
            ax.annotate(chr(65+i),item.position_mm[:2],xytext=(5,5),textcoords='offset points',fontsize=9)
        for frame in report['frames']:
            points=np.asarray(frame['antennae_mm']); ax.plot(points[:,0],points[:,1],'.-',color='#a3343b',lw=.7,ms=3)
        ax.set(xlim=(-1,12),ylim=(-6,6),xlabel='World x (mm)',ylabel='World y (mm)',title=f'Odor {chr(65+channel)} field')
        ax.set_aspect('equal'); ax.annotate('Body sensors',antennae[0,:2],xytext=(1.1,-4),arrowprops={'arrowstyle':'-','color':'#555555'},fontsize=9)
        box=ax.get_position(); colorbar=fig.colorbar(contour,cax=fig.add_axes([box.x0,.135,box.width,.016]),orientation='horizontal',ticks=[0,.25,.5])
        colorbar.ax.tick_params(labelsize=8,length=2)
    cases=report['geometry_counterfactuals']['cases']; data=np.asarray([c['observation']['sensory'] for c in cases])
    labels=['Distant sugar','Distant neutral','Touched sugar','Touched neutral','Odor removed','Odorless sugar']
    ax=axes[2]; ax.imshow(data,vmin=0,vmax=1,cmap='Greys',aspect='auto')
    ax.set_xticks(range(6),['A left','A right','B left','B right','Sugar','Contact'],rotation=35,ha='right')
    ax.set_yticks(range(6),labels); ax.set_title('Same measured body, changed stimulus')
    ax.tick_params(length=0); ax.axvline(3.5,color='#b66332',lw=1.5)
    for i in range(6):
        for j in range(6): ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',color='white' if data[i,j]>.6 else '#303531',fontsize=9)
    fig.suptitle('Food sensor calibration — before any model learns',x=.055,ha='left',y=.96,fontsize=18)
    fig.text(.055,.875,f'Actual FlyGym body origins at 0, 10 and 20 ms. Field slices at z = {height:.2f} mm; dimensionless odor inputs.',fontsize=10)
    fig.text(.055,.03,'Left: prescribed body motion in two engineered Gaussian fields. Right: separate contact probes at the initial measured pose.\nNo FLM policy, food approach, feeding or language transfer is measured. Distant sugar and neutral sources give identical sensory inputs.',fontsize=10)
    output=ROOT/'public/research/figures'; output.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'): fig.savefig(output/f'food-sensors.{extension}',dpi=140,metadata={'Date':None} if extension=='svg' else None)
    svg=output/'food-sensors.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
    plt.close(fig)
    with (output/'food-sensors.csv').open('w',newline='',encoding='utf8') as handle:
        writer=csv.writer(handle); writer.writerow(['case',*report['channels']]); writer.writerows((row['label'],*row['observation']['sensory']) for row in cases)
    files=[output/f'food-sensors.{extension}' for extension in ('png','svg','csv')]
    result=dict(source_record_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        coordinate_slice_z_mm=height,artifacts={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Field functions and recorded body-position geometry; no language or behavior-training result.')
    (ROOT/'reports/food-sensors/figure.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
