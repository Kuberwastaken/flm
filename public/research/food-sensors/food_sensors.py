"""Engineered odor fields and geometric food contact; no biological receptor model.

Coordinates are millimeters. Odor and sugar amplitudes are dimensionless task
units. Source labels, reward history and task phase never enter the six inputs.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np

CHANNELS = ('odor_a_left','odor_a_right','odor_b_left','odor_b_right','sugar_contact','source_contact')
ODOR_BODY_ORIGINS = ('l_funiculus','r_funiculus')
CONTACT_BODY_ORIGINS = tuple(f'{leg}_tarsus5' for leg in ('lf','lm','lh','rf','rm','rh'))


@dataclass(frozen=True)
class Source:
    name: str
    position_mm: tuple[float,float,float]
    spread_mm: tuple[float,float,float]
    odor: tuple[float,float]
    sugar: float
    contact_radius_mm: float
    contact_height_mm: float

    def validate(self):
        if not isinstance(self.name,str) or not self.name: raise ValueError('Source name required')
        for values,size in ((self.position_mm,3),(self.spread_mm,3),(self.odor,2)):
            if not isinstance(values,tuple) or len(values)!=size or any(type(v) not in (float,int) or not np.isfinite(v) for v in values):
                raise ValueError('Finite immutable source coordinates and amplitudes required')
        if any(v<=0 for v in self.spread_mm) or any(v<0 for v in self.odor):
            raise ValueError('Positive spreads and nonnegative odor amplitudes required')
        for value in (self.sugar,self.contact_radius_mm,self.contact_height_mm):
            if type(value) not in (float,int) or not np.isfinite(value): raise ValueError('Finite contact parameters required')
        if not 0<=self.sugar<=1 or self.contact_radius_mm<=0 or self.contact_height_mm<=0:
            raise ValueError('Sugar must be in [0,1]; contact radius and height positive')


def points(value, count=None):
    array=np.asarray(value,dtype=np.float64)
    if array.ndim!=2 or array.shape[1]!=3 or not len(array) or not np.isfinite(array).all() or count is not None and len(array)!=count:
        raise ValueError('Finite points with shape [count,3] required')
    return array


class FoodField:
    def __init__(self,sources):
        self.sources=tuple(sources)
        if not self.sources or any(type(source) is not Source for source in self.sources):
            raise ValueError('Require a nonempty source inventory')
        for source in self.sources: source.validate()
        if len({source.name for source in self.sources})!=len(self.sources): raise ValueError('Unique source names required')

    def card(self):
        return dict(format='flm-engineered-food-field-v1',sources=[asdict(s) for s in self.sources],channels=list(CHANNELS),
            coordinate_units='mm',odor_units='dimensionless, summed Gaussian fields; fixed c/(1+c) observation transform',
            sugar_units='dimensionless contact intensity in [0,1]; maximum over touched sources',
            geometry='Virtual upright closed cylinders from source z to z+height; contact uses six tarsus5 body-frame origins, not collision manifolds',
            scope='Synthetic sensor/task interface. No fluid dynamics, identified receptor neurons, ingestion, reward update or learned behavior.')

    def odor_at(self,positions):
        positions=points(positions); output=np.zeros((len(positions),2),dtype=np.float64)
        # Each field depends only on this source and physical location; no global
        # normalization by the nearest/rewarded source or by the current scene.
        for source in self.sources:
            with np.errstate(over='ignore'):
                distance=((positions-np.asarray(source.position_mm))/np.asarray(source.spread_mm))**2
                profile=np.exp(-.5*distance.sum(axis=1))
            output+=profile[:,None]*np.asarray(source.odor)[None,:]
        if not np.isfinite(output).all(): raise ValueError('Odor superposition overflow')
        return output

    def contact_at(self,positions):
        positions=points(positions); contact=np.zeros((len(positions),len(self.sources)),dtype=bool)
        for index,source in enumerate(self.sources):
            delta=positions-np.asarray(source.position_mm)
            contact[:,index]=(np.hypot(delta[:,0],delta[:,1])<=source.contact_radius_mm)&(delta[:,2]>=0)&(delta[:,2]<=source.contact_height_mm)
        return contact

    def observe(self,antennae_mm,contact_points_mm,*,missing_odor=False):
        if type(missing_odor) is not bool: raise ValueError('Explicit boolean odor intervention required')
        antennae=points(antennae_mm,2); contacts=points(contact_points_mm,6)
        raw=self.odor_at(antennae); sensed=np.zeros_like(raw) if missing_odor else raw
        # A fixed scale; retain tiny positive signals without subtractive cancellation.
        normalized=sensed/(1+sensed)
        touched=self.contact_at(contacts); source_contact=touched.any(axis=0)
        sugar=max((s.sugar for s,hit in zip(self.sources,source_contact) if hit),default=0.)
        vector=np.array([*normalized[:,0],*normalized[:,1],sugar,float(source_contact.any())],dtype=np.float64)
        vector.setflags(write=False)
        return dict(sensory=vector,raw_odor=raw,contact_mask=touched,
            diagnostic_contacted_sources=[s.name for s,hit in zip(self.sources,source_contact) if hit])


def body_sensor_positions(body_names,body_positions):
    """Read actual simulated body-frame origins in a checked canonical order."""
    names=list(body_names); positions=points(body_positions,len(names))
    if len(set(names))!=len(names): raise ValueError('Unique body names required')
    if not set((*ODOR_BODY_ORIGINS,*CONTACT_BODY_ORIGINS))<=set(names): raise ValueError('Required sensor body origins missing')
    lookup={name:positions[index] for index,name in enumerate(names)}
    return (np.array([lookup[name] for name in ODOR_BODY_ORIGINS]),
            np.array([lookup[name] for name in CONTACT_BODY_ORIGINS]))
