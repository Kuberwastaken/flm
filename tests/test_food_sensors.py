"""Analytical and intervention tests for the engineered sensor interface."""
from dataclasses import replace
import unittest
import numpy as np

from flm.food_sensors import FoodField, Source, CHANNELS, ODOR_BODY_ORIGINS, CONTACT_BODY_ORIGINS, body_sensor_positions


def source(name='a',**changes):
    return replace(Source(name,(0.,0.,0.),(2.,3.,4.),(1.,0.),1.,1.,.2),**changes)


class FoodSensorTests(unittest.TestCase):
    def test_gaussian_peak_decay_superposition_and_fixed_observation_scale(self):
        field=FoodField([source(),source('b',odor=(0.,2.))])
        actual=field.odor_at([[0,0,0],[2,3,4]])
        np.testing.assert_allclose(actual,[[1,2],[np.exp(-1.5),2*np.exp(-1.5)]],rtol=1e-14)
        observation=field.observe([[0,0,0],[0,0,0]],[[9,9,9]]*6)
        np.testing.assert_allclose(observation['sensory'],[.5,.5,2/3,2/3,0,0])
        self.assertEqual(len(CHANNELS),6); self.assertFalse(observation['sensory'].flags.writeable)
        self.assertGreater(field.observe([[20,0,0]]*2,[[9,9,9]]*6)['sensory'][0],0.)

    def test_contact_requires_both_lateral_and_vertical_geometry_with_closed_boundaries(self):
        points=[[0,0,0],[1,0,.2],[0,0,-.001],[0,0,.201],[1.001,0,0],[.7,.7,.1]]
        np.testing.assert_array_equal(FoodField([source()]).contact_at(points)[:,0],[True,True,False,False,False,True])

    def test_reward_reversal_is_invisible_before_contact_and_only_changes_taste_on_contact(self):
        original=FoodField([source()]); reversed_field=FoodField([source(sugar=0.)])
        antennae=[[1,1,1],[1,-1,1]]
        first=original.observe(antennae,[[4,4,4]]*6); second=reversed_field.observe(antennae,[[4,4,4]]*6)
        np.testing.assert_array_equal(first['sensory'],second['sensory'])
        first=original.observe(antennae,[[0,0,.1]]*6); second=reversed_field.observe(antennae,[[0,0,.1]]*6)
        np.testing.assert_array_equal(first['sensory'][:4],second['sensory'][:4])
        self.assertEqual(first['sensory'][4],1.); self.assertEqual(second['sensory'][4],0.)
        self.assertEqual(first['sensory'][5],second['sensory'][5])

    def test_missing_odor_neutral_and_odorless_sugar_remain_distinct(self):
        antennae=[[0,0,.1],[0,0,.1]]; contact=[[0,0,.1]]*6
        field=FoodField([source()]); dropped=field.observe(antennae,contact,missing_odor=True)
        np.testing.assert_array_equal(dropped['sensory'],[0,0,0,0,1,1])
        self.assertGreater(dropped['raw_odor'].sum(),0.)
        odorless=FoodField([source(odor=(0.,0.))]).observe(antennae,contact)
        np.testing.assert_array_equal(odorless['sensory'],dropped['sensory'])
        neutral=FoodField([source(odor=(0.,0.),sugar=0.)]).observe(antennae,contact)
        np.testing.assert_array_equal(neutral['sensory'],[0,0,0,0,0,1])

    def test_rigid_translation_rotation_and_sensor_swapping_preserve_physical_meaning(self):
        item=source(spread_mm=(2.,2.,4.)); field=FoodField([item])
        antennae=np.array([[.5,.2,.1],[.5,-.2,.1]]); contacts=np.tile([.5,.1,.1],(6,1))
        base=field.observe(antennae,contacts)['sensory']
        offset=np.array([3.,-5.,2.]); moved=FoodField([replace(item,position_mm=tuple(map(float,offset)))])
        np.testing.assert_allclose(base,moved.observe(antennae+offset,contacts+offset)['sensory'],rtol=1e-14)
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        np.testing.assert_allclose(base,field.observe(antennae@rotation.T,contacts@rotation.T)['sensory'],rtol=1e-14)
        swapped=field.observe(antennae[::-1],contacts)['sensory']
        np.testing.assert_array_equal(swapped,base[[1,0,3,2,4,5]])

    def test_source_names_order_and_body_order_cannot_change_sensory_input(self):
        first=source(); second=source('b',position_mm=(4.,0.,0.),odor=(0.,1.),sugar=.5)
        antennae=[[.2,.1,.1],[.2,-.1,.1]]; contacts=[[.1,0,.1]]*6
        a=FoodField([first,second]).observe(antennae,contacts)
        b=FoodField([replace(second,name='other'),replace(first,name='renamed')]).observe(antennae,contacts)
        np.testing.assert_array_equal(a['sensory'],b['sensory'])
        names=list((*ODOR_BODY_ORIGINS,*CONTACT_BODY_ORIGINS)); positions=np.arange(24).reshape(8,3)
        x=body_sensor_positions(names,positions); y=body_sensor_positions(names[::-1],positions[::-1])
        for left,right in zip(x,y): np.testing.assert_array_equal(left,right)

    def test_invalid_sources_positions_and_sensor_inventories_fail(self):
        for item in (source(spread_mm=(0.,2.,3.)),source(odor=(-1.,0.)),source(sugar=2.),
                     source(position_mm=(float('nan'),0.,0.)),source(contact_height_mm=0.)):
            with self.assertRaises(ValueError): FoodField([item])
        with self.assertRaises(ValueError): FoodField([source(),source()])
        with self.assertRaises(ValueError): body_sensor_positions(['missing'],[[0,0,0]])
        field=FoodField([source()])
        for points in ([],[[0,0]],[[0,0,float('inf')]]):
            with self.assertRaises(ValueError): field.odor_at(points)
        with self.assertRaises(ValueError): field.observe([[0,0,0]],[[0,0,0]]*6)
        with self.assertRaises(ValueError): field.observe([[0,0,0]]*2,[[0,0,0]]*5)
        with self.assertRaises(ValueError): field.observe([[0,0,0]]*2,[[0,0,0]]*6,missing_odor=1)


if __name__=='__main__': unittest.main()
