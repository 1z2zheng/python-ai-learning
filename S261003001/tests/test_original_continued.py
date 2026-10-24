import copy
import unittest
from pathlib import Path

import torch
from torch import nn

from classification.train_mixed import Record
from classification.train_original_continued import (
    conservative_loss, eligible, freeze_batchnorm, sampler_for,
)


class ContinuationChecks(unittest.TestCase):
    def test_a_better_combined_score_cannot_hide_accuracy_or_f1_regression(self):
        baseline = {"ps": {"accuracy": .8, "macro_f1_present": .75}, "pv": {"accuracy": .99}}
        candidate = copy.deepcopy(baseline)
        candidate["ps"].update(accuracy=.79, macro_f1_present=.90)
        self.assertFalse(eligible(candidate, baseline))
        candidate["ps"].update(accuracy=.90, macro_f1_present=.74)
        self.assertFalse(eligible(candidate, baseline))
        candidate["ps"].update(accuracy=.81, macro_f1_present=.76)
        self.assertTrue(eligible(candidate, baseline))
        candidate["pv"]["accuracy"] = .98
        self.assertFalse(eligible(candidate, baseline))

    def test_teacher_is_not_optimized_and_matching_distributions_have_no_anchor_cost(self):
        student = torch.tensor([[1., 2., 3.], [3., 2., 1.]], requires_grad=True)
        teacher = student.detach().clone().requires_grad_()
        labels = torch.tensor([2, 0])
        unanchored = conservative_loss(student, teacher, labels, 0)
        anchored = conservative_loss(student, teacher, labels, .7)
        self.assertAlmostEqual(unanchored.item(), anchored.item(), places=6)
        anchored.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_thousands_of_new_source_files_do_not_dominate_sampling(self):
        records = [Record(Path(f"pv{i}"), 0, "blackrot", "plantvillage") for i in range(4)]
        records += [Record(Path(f"ps{i}"), 0, "blackrot", "plantseg") for i in range(2)]
        records += [Record(Path(f"field{i}"), 0, "blackrot", "appleleaf9") for i in range(1000)]
        records += [Record(Path("healthy"), 1, "healthy", "plantvillage")]
        sampler = sampler_for(records, 20, 42)
        weights = sampler.weights
        total = weights.sum().item()
        self.assertAlmostEqual(weights[:-1].sum().item() / total, .5)
        new_mass = sum(weights[i].item() for i, r in enumerate(records) if r.domain == "appleleaf9")
        self.assertAlmostEqual(new_mass / weights[:-1].sum().item(), .125)

    def test_batchnorm_statistics_remain_fixed_during_training(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(.2))
        model.train()
        freeze_batchnorm(model)
        previous = model[1].running_mean.clone()
        model(torch.randn(8, 4))
        self.assertTrue(torch.equal(previous, model[1].running_mean))
        self.assertTrue(model[2].training)


if __name__ == "__main__":
    unittest.main()
