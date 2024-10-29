"""
Entity Matching Axiom Generator.

This module provides an axiom generation functionality for entity matching
tasks.
"""

import ltn
import numpy as np
import tensorflow as tf


class AxiomGenerator:
    """Axiom Generator Class.

    The class provides an axiom generator for entity matching tasks.
    """

    def __init__(self, data_generator):
        """Initialize an axiom generator object."""
        self.data_generator = data_generator

        self.Sim = ltn.Predicate.Lambda(
            lambda args: tf.exp(-1.0 * tf.norm(args[0] - args[1], axis=1, ord=np.inf))
        )
        self.Not = ltn.Wrapper_Connective(ltn.fuzzy_ops.Not_Std())
        self.And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
        self.Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
        self.Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())
        self.Equiv = ltn.Wrapper_Connective(
            ltn.fuzzy_ops.Equiv(
                ltn.fuzzy_ops.And_Prod(), ltn.fuzzy_ops.Implies_Reichenbach()
            )
        )
        self.ForAll = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2), semantics="forall"
        )
        self.Exists = ltn.Wrapper_Quantifier(
            ltn.fuzzy_ops.Aggreg_pMean(p=2), semantics="exists"
        )
        self.FormAgg = ltn.Wrapper_Formula_Aggregator(
            ltn.fuzzy_ops.Aggreg_pMeanError(p=2)
        )

    def __select_field_constants(self, value):
        examples = self.data_generator._DataGenerator__select_features(value)
        consts = [
            [ltn.Constant(point, trainable=False) for point in feature]
            for i, feature in enumerate(examples)
        ]
        return consts

    def field_matching_constants(self):
        """Return field matching constants."""
        return self.__select_field_constants(1.0)

    def field_non_matching_constants(self):
        """Return field non matching constants."""
        return self.__select_field_constants(0.0)

    def matching_axioms(self, record_classifier):
        """Return matching axioms."""
        fc = self.field_matching_constants()
        rc = [[f[i] for f in fc] for i in range(len(fc[0]))]
        field_predicates = [ltn.Predicate(f) for f in record_classifier.field_networks]
        record_predicate = ltn.Predicate(record_classifier)
        axioms = []
        for r in rc:
            stmt = field_predicates[0](r[0])
            for i, F in enumerate(field_predicates[1:]):
                stmt = self.Or(stmt, F(r[i + 1]))
            axioms.append(stmt)
            axioms.append(record_predicate(r))
        return axioms

    def non_matching_axioms(self, record_classifier):
        """Return non matching axioms."""
        fc = self.field_non_matching_constants()
        rc = [[f[i] for f in fc] for i in range(len(fc[0]))]
        field_predicates = [ltn.Predicate(f) for f in record_classifier.field_networks]
        record_predicate = ltn.Predicate(record_classifier)
        axioms = []
        for r in rc:
            stmt = field_predicates[0](r[0])
            for i, F in enumerate(field_predicates[1:]):
                stmt = self.And(stmt, F(r[i + 1]))
            axioms.append(self.Not(stmt))
            axioms.append(self.Not(record_predicate(r)))
        return axioms
