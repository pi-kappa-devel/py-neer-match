"""
Entity matching axiom generator module.

This module provides an axiom generation functionality for entity matching tasks.
"""

from neer_match.data_generator import DataGenerator
import ltn
import numpy as np
import tensorflow as tf
import typing


class AxiomGenerator:
    """Axiom generator class.

    The class contains the definitions for the logical connectives, functions,
    quantifiers, and aggregators used in the axiom generation process.

    Attributes:
        data_generator (DataGenerator): A data generator object.
        Sim (ltn.Predicate.Lambda): A similarity predicate.
        Not (ltn.Wrapper_Connective): A negation connective.
        And (ltn.Wrapper_Connective): A conjunction connective.
        Or (ltn.Wrapper_Connective): A disjunction connective.
        Implies (ltn.Wrapper_Connective): An implication connective.
        Equiv (ltn.Wrapper_Connective): An equivalence connective.
        ForAll (ltn.Wrapper_Quantifier): A universal quantifier.
        Exists (ltn.Wrapper_Quantifier): An existential quantifier.
        FormAgg (ltn.Wrapper_Formula_Aggregator): A formula aggregator.
    """

    def __init__(self, data_generator: DataGenerator) -> None:
        r"""Initialize an axiom generator object.

        By default, the axiom generator uses product t-norm and co t-norm for
        conjunction and disjunction, respectively. In addition, it uses Reichenbach
        implications. The quantifiers are based on the p-mean error and p-mean
        aggregators (see also :ltn-tut-2c:`\ `).

        Args:
            data_generator: A data generator object.
        """
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

    def __select_field_constants(
        self, value: float
    ) -> typing.List[typing.List[ltn.Constant]]:
        examples = self.data_generator._DataGenerator__select_features(value)
        consts = [
            [ltn.Constant(point, trainable=False) for point in feature]
            for i, feature in enumerate(examples)
        ]
        return consts

    def field_matching_constants(self) -> typing.List[typing.List[ltn.Constant]]:
        """Return field matching constants.

        The outer list contains contains matching examples and the inner lists contain
        the constants of the features.
        """
        return self.__select_field_constants(1.0)

    def field_non_matching_constants(self) -> typing.List[typing.List[ltn.Constant]]:
        """Return field non matching constants.

        The outer list contains contains non-matching examples and the inner lists
        contain the constants of the features.
        """
        return self.__select_field_constants(0.0)
