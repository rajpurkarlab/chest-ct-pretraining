import pandas as pd
import unittest
import os
import sys

sys.path.append(os.getcwd())

from pandas.core.algorithms import unique

from pe_models.constants import *


class RSNADatasetTestCase(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(RSNA_TRAIN_CSV)

        self.df_train = self.df[self.df[RSNA_SPLIT_COL] == "train"]
        self.df_val = self.df[self.df[RSNA_SPLIT_COL] == "valid"]
        self.df_test = self.df[self.df[RSNA_SPLIT_COL] == "test"]

        self.df_ins_train = self.df[self.df[RSNA_INSTITUTION_SPLIT_COL] == "train"]
        self.df_ins_val = self.df[self.df[RSNA_INSTITUTION_SPLIT_COL] == "valid"]
        self.df_ins_test = self.df[self.df[RSNA_INSTITUTION_SPLIT_COL] == "test"]
        self.df_ins_stanford = self.df[
            self.df[RSNA_INSTITUTION_SPLIT_COL] == "stanford_test"
        ]

    def test_num_study_in_split_gt_zero(self):

        num_train_studies = self.df_train[RSNA_STUDY_COL].nunique()
        num_val_studies = self.df_val[RSNA_STUDY_COL].nunique()
        num_test_studies = self.df_test[RSNA_STUDY_COL].nunique()

        num_ins_train_studies = self.df_ins_train[RSNA_STUDY_COL].nunique()
        num_ins_val_studies = self.df_ins_val[RSNA_STUDY_COL].nunique()
        num_ins_test_studies = self.df_ins_test[RSNA_STUDY_COL].nunique()
        num_ins_stanford_studies = self.df_ins_stanford[RSNA_STUDY_COL].nunique()

        self.assertEqual(num_train_studies, 5095, "no studies in training set")
        self.assertEqual(num_val_studies, 1092, "no studies in val set")
        self.assertEqual(num_test_studies, 1092, "no studies in test set")
        self.assertEqual(
            num_ins_train_studies, 4620, "no studies in institutional training set"
        )
        self.assertEqual(
            num_ins_val_studies, 990, "no studies in institutional val set"
        )
        self.assertEqual(
            num_ins_test_studies, 991, "no studies in institutional test set"
        )
        self.assertEqual(
            num_ins_stanford_studies, 678, "no studies in institutional stanford set"
        )

    def test_no_overlap_study_in_split(self):

        train_studies = self.df_train[RSNA_STUDY_COL].unique()
        val_studies = self.df_val[RSNA_STUDY_COL].unique()
        test_studies = self.df_test[RSNA_STUDY_COL].unique()

        self.assertFalse(
            bool(set(train_studies) & set(val_studies)), "train and val overlap"
        )
        self.assertFalse(
            bool(set(train_studies) & set(test_studies)), "train and test overlap"
        )
        self.assertFalse(
            bool(set(val_studies) & set(test_studies)), "val and test overlap"
        )

    def test_no_overlap_study_in_institution_split(self):

        train_studies = self.df_ins_train[RSNA_STUDY_COL].unique()
        val_studies = self.df_ins_val[RSNA_STUDY_COL].unique()
        test_studies = self.df_ins_test[RSNA_STUDY_COL].unique()
        stanford_studies = self.df_ins_stanford[RSNA_STUDY_COL].unique()

        self.assertFalse(
            bool(set(train_studies) & set(val_studies)), "train and val overlap"
        )
        self.assertFalse(
            bool(set(train_studies) & set(test_studies)), "train and test overlap"
        )
        self.assertFalse(
            bool(set(val_studies) & set(test_studies)), "val and test overlap"
        )
        self.assertFalse(
            bool(set(stanford_studies) & set(test_studies)), "test and stanford overlap"
        )
        self.assertFalse(
            bool(set(stanford_studies) & set(train_studies)),
            "train and stanford overlap",
        )
        self.assertFalse(
            bool(set(stanford_studies) & set(val_studies)), "val and stanford overlap"
        )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
