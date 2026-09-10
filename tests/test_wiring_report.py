import copy
import unittest
import numpy as np
from flm.wiring_report import paired_statistics, validate_panels


class WiringReportTests(unittest.TestCase):
    def report(self):
        labels=np.tile([0,1],128); probabilities=np.full((256,2),.25)
        probabilities[np.arange(256),labels]=.75
        panels=[dict(delay=d,episodes=256,labels=labels.tolist(),action_probabilities=probabilities.tolist(),
            scores=dict(original=dict(accuracy=1.,cross_entropy=-np.log(.75)),reversed=dict(accuracy=0.,cross_entropy=-np.log(.25)))) for d in (4,8,12,24,48)]
        return dict(probes=[dict(step=s,current_mapping='reversed' if 300<s<=600 else 'original',panels=copy.deepcopy(panels)) for s in range(0,901,100)])

    def test_recomputes_accuracy_and_rejects_missing_panels_or_changed_scores(self):
        report=self.report(); validate_panels(report)
        bad=copy.deepcopy(report); bad['probes'][0]['panels'][0]['scores']['original']['accuracy']=.5
        with self.assertRaisesRegex(ValueError,'Accuracy differs'): validate_panels(bad)
        bad=copy.deepcopy(report); bad['probes'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete checkpoint'): validate_panels(bad)
        bad=copy.deepcopy(report); bad['probes'][1]['panels'][2]['action_probabilities'][0][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid action'): validate_panels(bad)

    def test_pairing_retains_seed_sign_changes(self):
        result=paired_statistics([1.,0.,1.],[0.,1.,1.])
        self.assertEqual(result['values'],[1.,-1.,0.]); self.assertEqual(result['mean'],0.)
        self.assertEqual(result['minimum'],-1.); self.assertEqual(result['maximum'],1.)
        with self.assertRaisesRegex(ValueError,'Unpaired'): paired_statistics([1.,0.],[0.,1.])


if __name__=='__main__': unittest.main()
