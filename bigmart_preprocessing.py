"""Leakage-free preprocessing components for the BigMart sales model.

These transformers live in a module rather than inside the notebook for two reasons:

1. **Deployability.** ``joblib`` pickles a *reference* to a class (module path +
   name), not the class's source code. A transformer defined in the notebook's
   ``__main__`` namespace cannot be unpickled by ``streamlit_app.py``, so the exported
   pipeline would fail to load in the Streamlit app.
2. **Correctness.** Putting every cleaning and imputation rule *inside* the
   scikit-learn pipeline means each rule is re-fitted on the training rows of
   every cross-validation fold. Cleaning the full DataFrame up-front instead
   would let test-fold statistics leak into training, quietly flattering every
   score the notebook reports.

The split of responsibilities below is deliberate:

* :class:`DomainCleaner` applies rules that are *deterministic and row-wise* --
  they learn nothing from the data, so they cannot leak and need no fitting.
* :class:`GroupStatisticImputer` learns a statistic per group and therefore
  **must** be fitted on training data only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

#: Year the BigMart transactions are taken to have been recorded, used to turn
#: ``Outlet_Establishment_Year`` into the more interpretable ``Outlet_Age``.
REFERENCE_YEAR = 2013

#: The raw data records the same two fat categories under five spellings.
FAT_CONTENT_MAP = {
    "Low Fat": "Low Fat",
    "low fat": "Low Fat",
    "LF": "Low Fat",
    "Regular": "Regular",
    "reg": "Regular",
}

#: Item types whose ``Item_Identifier`` always carries the ``NC`` (non-consumable)
#: prefix. Verified empirically in the notebook: the mapping is exact, which is
#: what lets the "Non-Edible" fix run at serving time, where no identifier exists.
NON_CONSUMABLE_ITEM_TYPES = frozenset({"Health and Hygiene", "Household", "Others"})


class DomainCleaner(BaseEstimator, TransformerMixin):
    """Apply the deterministic data-quality rules diagnosed during EDA.

    Every rule here is a fixed row-wise mapping, so ``fit`` is a no-op and no
    information can cross the train/test boundary:

    * collapse the five ``Item_Fat_Content`` spellings into two categories;
    * relabel non-consumable goods as ``Non-Edible`` -- fat content is a
      meaningless attribute for household cleaners and hygiene products, and
      leaving them as "Low Fat" feeds the model a spurious diet signal;
    * convert ``Item_Visibility == 0`` to ``NaN``, since a stocked item cannot
      occupy zero shelf area (these are unrecorded values, not measurements)
      -- the actual imputation is a separate, *fitted* step;
    * derive ``Outlet_Age`` from ``Outlet_Establishment_Year``.

    Columns that are absent from ``X`` are skipped, so the same transformer
    serves both the 12-column training frame and the 9-column payload the
    Streamlit app sends.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DomainCleaner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        if "Item_Fat_Content" in X.columns:
            # `.replace` rather than `.map` so values already in canonical form
            # (e.g. "Non-Edible" coming from the app) survive untouched instead
            # of silently becoming NaN.
            X["Item_Fat_Content"] = X["Item_Fat_Content"].replace(FAT_CONTENT_MAP)

            if "Item_Type" in X.columns:
                is_non_consumable = X["Item_Type"].isin(NON_CONSUMABLE_ITEM_TYPES)
                X.loc[is_non_consumable, "Item_Fat_Content"] = "Non-Edible"

        if "Item_Visibility" in X.columns:
            X["Item_Visibility"] = X["Item_Visibility"].replace(0, np.nan)

        if "Outlet_Establishment_Year" in X.columns:
            X["Outlet_Age"] = REFERENCE_YEAR - X["Outlet_Establishment_Year"]

        return X


class GroupStatisticImputer(BaseEstimator, TransformerMixin):
    """Fill missing values in ``column`` with a statistic learned per ``group_by`` group.

    The lookup table and the global fallback are both computed in :meth:`fit`,
    i.e. from training rows only. Groups unseen at fit time (and groups whose
    every training value was missing) fall back to the global statistic.

    Parameters
    ----------
    column:
        Column whose missing values are to be filled.
    group_by:
        Column defining the groups. If it is not present in ``X`` at transform
        time -- as happens when the Streamlit app omits ``Item_Identifier`` --
        the global fallback is used instead of raising.
    strategy:
        ``"mean"`` for numeric columns, ``"mode"`` for categorical ones.
    """

    def __init__(self, column: str, group_by: str, strategy: str = "mean"):
        self.column = column
        self.group_by = group_by
        self.strategy = strategy

    def _aggregate(self, series: pd.Series):
        if self.strategy == "mean":
            return series.mean()
        if self.strategy == "mode":
            modes = series.mode()
            return modes.iloc[0] if len(modes) else np.nan
        raise ValueError(f"Unknown strategy {self.strategy!r}; expected 'mean' or 'mode'.")

    def fit(self, X: pd.DataFrame, y=None) -> "GroupStatisticImputer":
        known = X.dropna(subset=[self.column])
        self.group_statistic_ = (
            known.groupby(self.group_by)[self.column].agg(self._aggregate)
            if self.group_by in X.columns
            else pd.Series(dtype="object")
        )
        self.global_statistic_ = self._aggregate(known[self.column])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        missing = X[self.column].isna()
        if not missing.any():
            return X

        if self.group_by in X.columns and len(self.group_statistic_):
            from_group = X.loc[missing, self.group_by].map(self.group_statistic_)
            X.loc[missing, self.column] = from_group.to_numpy()

        # Anything still missing: unseen group, absent group column, or a group
        # whose training values were all missing.
        X[self.column] = X[self.column].fillna(self.global_statistic_)
        return X
