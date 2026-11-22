// Runs the exported pit and compound models in the browser (twin of export_model.py).
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Model = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function vector(model, row) {
    return model.inputs.map((spec) => {
      if ("cat" in spec) return String(row[spec.cat]) === spec.value ? 1 : 0;
      const v = row[spec.num];
      if (v === null || v === undefined || v === "" || Number.isNaN(Number(v))) return NaN;
      return typeof v === "boolean" ? (v ? 1 : 0) : Number(v);
    });
  }

  function treeValue(t, x) {
    let node = t.root[0];
    while (node >= 0) {
      let v = x[t.f[node]];
      const miss = t.m[node];
      if (Number.isNaN(v) && miss !== 2) v = 0;
      if ((miss === 2 && Number.isNaN(v)) || (miss === 1 && Math.abs(v) <= 1e-35)) {
        node = t.d[node] ? t.l[node] : t.r[node];
      } else {
        node = v <= t.t[node] ? t.l[node] : t.r[node];
      }
    }
    return t.v[~node];
  }

  function raw(model, row) {
    const x = vector(model, row);
    const sums = new Array(model.num_class).fill(0);
    model.trees.forEach((t, i) => { sums[i % model.num_class] += treeValue(t, x); });
    return sums;
  }

  function interp(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    if (xs[hi] === xs[lo]) return ys[hi];
    return ys[lo] + ((ys[hi] - ys[lo]) * (x - xs[lo])) / (xs[hi] - xs[lo]);
  }

  function pitProbability(models, row) {
    const pit = models.pit;
    const p = 1 / (1 + Math.exp(-raw(pit, row)[0]));
    return interp(p, pit.calibration.x, pit.calibration.y);
  }

  function compoundProbabilities(models, row) {
    const c = models.compound;
    const r = raw(c, row);
    const m = Math.max(...r);
    const e = r.map((v) => Math.exp(v - m));
    const total = e.reduce((a, b) => a + b, 0);
    return Object.fromEntries(c.classes.map((k, i) => [k, e[i] / total]));
  }

  return { pitProbability, compoundProbabilities };
});
