// Strategy engine, the browser twin of tire_strategy/strategy/engine.py.
// Same cost model and the same dynamic programme, so the site and Python agree
// (checked by tests/test_engine_parity.py). See the Python module for the maths.
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Engine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  const DRY = ["SOFT", "MEDIUM", "HARD"];
  const WET = ["INTERMEDIATE", "WET"];
  const WET_ON_DRY = { offset: 3.0, degradation: 0.12, life: 30 };
  const MARGIN_S = 0.5;
  const WINDOW_S = 2.0;
  const r2 = (x) => Math.round(x * 100) / 100;

  function circuitParams(params, circuit) {
    const base = params.default;
    const c = params.circuits[circuit || ""] || {};
    return {
      circuit,
      known: Object.keys(c).length > 0,
      race_laps: c.race_laps ?? null,
      degradation: { ...base.degradation, ...(c.degradation || {}) },
      offsets: { ...base.offsets, ...(c.offsets || {}) },
      tyre_life: { ...base.tyre_life, ...(c.tyre_life || {}) },
      pit_loss: c.pit_loss ?? base.pit_loss,
      p_caution: c.p_caution ?? base.p_caution,
      p_safety_car: c.p_safety_car ?? base.p_safety_car,
      caution_per_lap: c.caution_per_lap ?? base.caution_per_lap,
      sc_pit_factor: params.sc_pit_factor ?? 0.5,
      vsc_pit_factor: params.vsc_pit_factor ?? 0.65,
    };
  }

  function compound(p, c) {
    if (WET.includes(c)) return [WET_ON_DRY.offset, WET_ON_DRY.degradation, WET_ON_DRY.life];
    return [Number(p.offsets[c]), Number(p.degradation[c]), Math.trunc(p.tyre_life[c])];
  }

  function stintCost(p, c, laps, age0 = 0) {
    const [off, deg] = compound(p, c);
    return laps * off + deg * (laps * age0 + (laps * (laps - 1)) / 2);
  }

  function lapCost(p, c, age) {
    const [off, deg] = compound(p, c);
    return off + deg * age;
  }

  function bestLengths(p, seq, total, firstAge) {
    const k = seq.length;
    const caps = seq.map((c, i) => Math.min(total, compound(p, c)[2] - (i === 0 ? firstAge : 0)));
    if (caps.some((x) => x < 1) || caps.reduce((a, b) => a + b, 0) < total || total < k) return null;
    const f = Array.from({ length: k + 1 }, () => new Array(total + 1).fill(Infinity));
    const choice = Array.from({ length: k + 1 }, () => new Array(total + 1).fill(0));
    f[0][0] = 0;
    for (let i = 0; i < k; i++) {
      const age0 = i === 0 ? firstAge : 0;
      for (let n = 0; n <= total; n++) {
        const base = f[i][n];
        if (base === Infinity) continue;
        const maxLen = Math.min(caps[i], total - n);
        for (let len = 1; len <= maxLen; len++) {
          const cost = base + stintCost(p, seq[i], len, age0);
          if (cost < f[i + 1][n + len] - 1e-12) {
            f[i + 1][n + len] = cost;
            choice[i + 1][n + len] = len;
          }
        }
      }
    }
    if (f[k][total] === Infinity) return null;
    const lengths = [];
    let n = total;
    for (let i = k; i > 0; i--) {
      const len = choice[i][n];
      lengths.push(len);
      n -= len;
    }
    return [f[k][total], lengths.reverse()];
  }

  function* product(items, repeat) {
    if (repeat === 0) { yield []; return; }
    for (const head of items) for (const rest of product(items, repeat - 1)) yield [head, ...rest];
  }

  function* sequences(compounds, maxStops, first) {
    for (let stops = 0; stops <= maxStops; stops++) {
      if (first == null) yield* product(compounds, stops + 1);
      else for (const rest of product(compounds, stops)) yield [first, ...rest];
    }
  }

  function ruleOk(seq, used, rule) {
    if (!rule) return true;
    const all = [...used, ...seq];
    const dry = new Set(all.filter((c) => DRY.includes(c)));
    return dry.size >= 2 || all.some((c) => WET.includes(c));
  }

  function plan(p, totalLaps, opts = {}) {
    const { startLap = 1, current = null, used = [], compounds = DRY, twoCompoundRule = true,
      maxStops = 3, pitLoss = null } = opts;
    const remaining = totalLaps - startLap + 1;
    if (remaining < 1) return [];
    const loss = Number(pitLoss == null ? p.pit_loss : pitLoss);
    const [first, firstAge] = current ? current : [null, 0];
    const options = [];
    for (const seq of sequences(compounds, maxStops, first)) {
      if (!ruleOk(seq, used, twoCompoundRule)) continue;
      const best = bestLengths(p, seq, remaining, firstAge);
      if (!best) continue;
      const [cost, lengths] = best;
      const stops = seq.length - 1;
      const stints = [];
      let lap = startLap;
      seq.forEach((c, i) => {
        stints.push({ compound: c, laps: lengths[i], start_lap: lap, end_lap: lap + lengths[i] - 1 });
        lap += lengths[i];
      });
      options.push({
        stops, stints, pit_laps: stints.slice(0, -1).map((s) => s.end_lap),
        tyre_cost: r2(cost), pit_cost: r2(stops * loss), cost: r2(cost + stops * loss),
      });
    }
    options.sort((a, b) => a.cost - b.cost || a.stops - b.stops);
    if (options.length) {
      const best = options[0].cost;
      for (const o of options) o.delta = r2(o.cost - best);
    }
    return options;
  }

  function pitWindows(p, option, windowS = WINDOW_S) {
    const st = option.stints;
    const out = [];
    for (let i = 0; i < st.length - 1; i++) {
      const a = st[i], b = st[i + 1];
      const base = stintCost(p, a.compound, a.laps) + stintCost(p, b.compound, b.laps);
      let lo = a.end_lap, hi = a.end_lap;
      for (const dir of [-1, 1]) {
        for (let shift = 1; ; shift++) {
          const la = a.laps + dir * shift, lb = b.laps - dir * shift;
          if (la < 1 || lb < 1) break;
          if (la > compound(p, a.compound)[2] || lb > compound(p, b.compound)[2]) break;
          const cost = stintCost(p, a.compound, la) + stintCost(p, b.compound, lb);
          if (cost - base > windowS) break;
          if (dir < 0) lo = a.end_lap - shift; else hi = a.end_lap + shift;
        }
      }
      out.push([lo, hi]);
    }
    return out;
  }

  const label = (o) => o.stints.map((s) => s.compound[0]).join("-");

  function bestByStops(options) {
    const seen = new Map();
    for (const o of options) if (!seen.has(o.stops)) seen.set(o.stops, o);
    return [...seen.keys()].sort((a, b) => a - b).map((k) => seen.get(k));
  }

  function scaled(p, factor) {
    const deg = {};
    for (const [c, v] of Object.entries(p.degradation)) deg[c] = v * factor;
    return { ...p, degradation: deg };
  }

  function racePlan(p, totalLaps, opts = {}) {
    const { compounds = DRY, twoCompoundRule = true, maxStops = 3 } = opts;
    const options = plan(p, totalLaps, { compounds, twoCompoundRule, maxStops });
    if (!options.length) return { options: [], best: null };
    const top = bestByStops(options);
    for (const o of top) { o.windows = pitWindows(p, o); o.label = label(o); }
    const sensitivity = {};
    for (const [name, factor] of [["degradation_low", 0.75], ["degradation_high", 1.25]]) {
      const alt = plan(scaled(p, factor), totalLaps, { compounds, twoCompoundRule, maxStops });
      sensitivity[name] = alt.length ? { label: label(alt[0]), stops: alt[0].stops } : null;
    }
    const alternatives = options.slice(0, 8);
    for (const o of alternatives) if (!o.label) o.label = label(o);
    return { best: options[0], by_stops: top, top: alternatives, sensitivity };
  }

  function advise(p, s) {
    const { total_laps, lap, compound: tyre, tyre_age, caution = "none", rain = false,
      gap_ahead = null, gap_behind = null, compounds = DRY, two_compound_rule = true } = s;
    const remaining = total_laps - lap + 1;
    if (remaining < 1) throw new Error("The race is already over at that lap.");
    const used = new Set([...(s.used || []), tyre]);
    const factor = caution === "sc" ? p.sc_pit_factor : caution === "vsc" ? p.vsc_pit_factor : 1;
    const pitNow = r2(Number(p.pit_loss) * factor);
    const notes = [];
    if (rain) {
      const onSlicks = DRY.includes(tyre);
      return {
        decision: onSlicks ? "box" : "stay", compound: onSlicks ? "INTERMEDIATE" : null,
        source: "wet rule", delta: null, pit_loss_now: pitNow, stay: null, box: [],
        notes: [onSlicks
          ? "Rain: on slicks, fit intermediates (the rule teams follow almost always)."
          : "Rain: stay on wet-weather tyres."],
      };
    }
    const stayOpts = plan(p, total_laps, { startLap: lap, current: [tyre, tyre_age], used: [...used],
      compounds, twoCompoundRule: two_compound_rule });
    const stay = stayOpts[0] || null;
    const box = [];
    for (const c of compounds) {
      const opts = plan(p, total_laps, { startLap: lap, current: [c, 0], used: [...used, c],
        compounds, twoCompoundRule: two_compound_rule });
      if (opts.length) {
        const o = { ...opts[0] };
        o.cost = r2(o.cost + pitNow);
        o.pit_cost = r2(o.pit_cost + pitNow);
        o.compound = c;
        box.push(o);
      }
    }
    box.sort((a, b) => a.cost - b.cost);
    const bestBox = box[0] || null;
    if (!stay && !bestBox) throw new Error("No legal strategy from this state.");
    let decision, delta = null;
    if (!stay) {
      decision = "box";
      notes.push("This tyre cannot reach the flag or the rules require another compound.");
    } else if (!bestBox) {
      decision = "stay";
    } else {
      delta = r2(stay.cost - bestBox.cost);
      decision = delta > MARGIN_S ? "box" : delta < -MARGIN_S ? "stay" : "either";
    }
    if ((caution === "sc" || caution === "vsc") && bestBox) {
      const saved = Number(p.pit_loss) - pitNow;
      notes.push(`${caution.toUpperCase()} now: a stop costs ${pitNow.toFixed(1)}s instead of ` +
        `${Number(p.pit_loss).toFixed(1)}s (${saved.toFixed(1)}s cheaper).`);
    }
    if (bestBox && gap_behind != null && gap_behind < pitNow) {
      notes.push(`Boxing loses about ${pitNow.toFixed(0)}s: you would rejoin behind the car ` +
        `now ${gap_behind.toFixed(1)}s behind you.`);
    }
    if (bestBox && gap_ahead != null && caution === "none") {
      const gain = lapCost(p, tyre, tyre_age) - lapCost(p, bestBox.compound, 0);
      if (gain > 0 && 2 * gain >= gap_ahead) {
        notes.push(`Undercut on: fresh ${bestBox.compound.toLowerCase()}s are about ` +
          `${gain.toFixed(1)}s a lap faster and the car ahead is ${gap_ahead.toFixed(1)}s away.`);
      }
    }
    if (stay && stay.pit_laps.length) notes.push(`If you stay out, the best next stop is lap ${stay.pit_laps[0]}.`);
    return {
      decision, compound: bestBox && decision !== "stay" ? bestBox.compound : null,
      source: "strategy engine", delta, pit_loss_now: pitNow, stay, box, notes,
    };
  }

  return { DRY, circuitParams, plan, racePlan, advise, stintCost, lapCost, pitWindows };
});
