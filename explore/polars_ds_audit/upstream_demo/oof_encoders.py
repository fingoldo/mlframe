"""OOF wrappers над polars_ds.pipeline.Blueprint для target_encode / woe_encode.

**Зачем:** В polars_ds 0.10.3 target_encode и woe_encode фитятся на всём train'е,
что означает ЦЕЛЕВУЮ УТЕЧКУ на train — модель видит средние y по категориям,
в которые включены именно те же y. sklearn.preprocessing.TargetEncoder (1.3+)
решает это через встроенный cross-fitting (cv=5). category_encoders не
встраивает OOF, но WOEEncoder предлагает `randomized=True, sigma=0.05` —
лёгкая гауссовская регуляризация при fit_transform на train.

**Этот модуль = прототип upstream-патча.** Когда добавим `cv=` в polars_ds,
эти классы можно будет удалить. Пока — работают поверх текущего API 0.10.3.

Стратегии:
- `cv>0`  — K-fold, fit на out-of-fold, transform на in-fold.
- `cv=0`  — plain fit_transform (как polars_ds сейчас, эквивалент sklearn fit/transform).
- `randomized=True, sigma=s` — WOEEncoder-style: домножить закодированное значение
   на N(1, s) на train (не test). Более дешёвая альтернатива OOF без K перефитов.

Контракт:
- `fit(X_train, y_train)`: хранит финальный mapping (fit на ВСЁМ train) для применения к test.
- `transform(X)`: применяет финальный mapping. На test — без OOF, на train — детерминированно.
- `fit_transform(X_train, y_train)`: возвращает OOF-кодирование train (каждая строка закодирована
  mapping'ом, обученным БЕЗ её fold'а).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from polars_ds.pipeline import Blueprint
from sklearn.model_selection import KFold, StratifiedKFold


_TARGET_COL = "__y_oof__"


def _fit_encoder_bp(
    kind: Literal["target_encode", "woe_encode"],
    df: pl.DataFrame,
    cols: list[str],
    target: str,
    **kwargs,
) -> "object":
    """Строит и materialize'ит Blueprint для кодирования `cols` по `target`."""
    bp = Blueprint(df, name=kind, target=target)
    bp = getattr(bp, kind)(cols=cols, **kwargs)
    return bp.materialize()


class OOFEncoder:
    """K-fold OOF wrapper вокруг polars_ds target_encode/woe_encode.

    Parameters
    ----------
    kind : 'target_encode' | 'woe_encode'
    cols : список колонок для кодирования (string/categorical)
    cv : количество фолдов для OOF на train. 0 → plain fit_transform (небезопасно!)
    stratified : StratifiedKFold vs KFold
    random_state : seed
    randomized : если True и cv==0 — гауссовский шум на выходе train (WOEEncoder-style)
    sigma : std шума при randomized=True
    **pds_kwargs : пробрасываются в Blueprint.target_encode / .woe_encode (e.g. smoothing, min_samples_leaf)
    """

    def __init__(
        self,
        kind: Literal["target_encode", "woe_encode"] = "target_encode",
        cols: list[str] | None = None,
        cv: int = 5,
        stratified: bool = True,
        random_state: int = 0,
        randomized: bool = False,
        sigma: float = 0.05,
        **pds_kwargs,
    ):
        self.kind = kind
        self.cols = cols
        self.cv = cv
        self.stratified = stratified
        self.random_state = random_state
        self.randomized = randomized
        self.sigma = sigma
        self.pds_kwargs = pds_kwargs
        self._pipeline = None  # final encoder for transform()

    # ---- sklearn-совместимый API ----
    def fit(self, X: pl.DataFrame, y: pl.Series | np.ndarray | list) -> "OOFEncoder":
        """Финальный энкодер = fit на всём train. Используется для transform(test)."""
        X = self._ensure_pl(X)
        X2 = X.with_columns(pl.Series(_TARGET_COL, np.asarray(y, dtype=np.float64)))
        self._pipeline = _fit_encoder_bp(self.kind, X2, self.cols, _TARGET_COL, **self.pds_kwargs)
        self._target = _TARGET_COL
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """Применение финального энкодера (для test)."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() first")
        X = self._ensure_pl(X)
        # подсовываем фиктивный target-колонку, чтобы pipeline прогнался:
        needs_target = _TARGET_COL not in X.columns
        if needs_target:
            X = X.with_columns(pl.lit(0.0).alias(_TARGET_COL))
        out = self._pipeline.transform(X)
        if needs_target:
            out = out.drop(_TARGET_COL)
        return out

    def fit_transform(self, X: pl.DataFrame, y) -> pl.DataFrame:
        """OOF-кодирование train. Это главный метод ради leak-safety."""
        X = self._ensure_pl(X)
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(X)
        X2 = X.with_columns(pl.Series(_TARGET_COL, y_arr))

        if self.cv and self.cv > 1:
            encoded = {c: np.zeros(n, dtype=np.float64) for c in self.cols}
            if self.stratified:
                splitter = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                folds = list(splitter.split(np.zeros(n), y_arr.astype(int)))
            else:
                splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                folds = list(splitter.split(np.zeros(n)))
            for fold_idx, (oof_idx, in_idx) in enumerate(folds):
                # fit на OOF, transform на in-fold
                oof_df = X2[oof_idx.tolist()]
                in_df = X2[in_idx.tolist()]
                pipe = _fit_encoder_bp(self.kind, oof_df, self.cols, _TARGET_COL, **self.pds_kwargs)
                in_t = pipe.transform(in_df)
                for c in self.cols:
                    encoded[c][in_idx] = in_t[c].to_numpy()
            # финальный энкодер (на всём train) — для будущих transform(test)
            self._pipeline = _fit_encoder_bp(self.kind, X2, self.cols, _TARGET_COL, **self.pds_kwargs)
            # собираем выход
            out = X.clone()
            for c in self.cols:
                out = out.with_columns(pl.Series(c, encoded[c]))
            return out
        else:
            # plain fit_transform (leaky!) — совместимость с текущим polars_ds
            self._pipeline = _fit_encoder_bp(self.kind, X2, self.cols, _TARGET_COL, **self.pds_kwargs)
            out = self._pipeline.transform(X2).drop(_TARGET_COL)
            if self.randomized and self.sigma > 0:
                rng = np.random.default_rng(self.random_state)
                for c in self.cols:
                    vals = out[c].to_numpy()
                    vals = vals * rng.normal(1.0, self.sigma, size=vals.shape)
                    out = out.with_columns(pl.Series(c, vals))
            return out

    @staticmethod
    def _ensure_pl(X) -> pl.DataFrame:
        if isinstance(X, pl.DataFrame):
            return X
        return pl.DataFrame(X)


# Удобные алиасы
class OOFTargetEncoder(OOFEncoder):
    def __init__(self, **kw):
        kw.setdefault("kind", "target_encode")
        super().__init__(**kw)


class OOFWOEEncoder(OOFEncoder):
    def __init__(self, **kw):
        kw.setdefault("kind", "woe_encode")
        super().__init__(**kw)
