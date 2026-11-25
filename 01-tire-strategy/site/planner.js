// Strategy planner and the next-race prediction.
"use strict";

const Planner = (() => {
  function params(circuit, overrides = {}) {
    const p = Engine.circuitParams(App.data.params, circuit);
    if (overrides.pitLoss) p.pit_loss = overrides.pitLoss;
    if (overrides.wear && overrides.wear !== 1) {
      const deg = {};
      for (const [c, v] of Object.entries(p.degradation)) deg[c] = v * overrides.wear;
      p.degradation = deg;
    }
    return p;
  }

  function facts(p, circuit) {
    const c = App.data.params.circuits[circuit] || {};
    const fmt = (v) => (v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2));
    const rows = ["SOFT", "MEDIUM", "HARD"].map((k) => el("tr", {},
      el("th", { scope: "row" }, chip(LONG[k]), NAMES[LONG[k]]),
      el("td", { text: `${fmt(p.offsets[k])} s` }),
      el("td", { text: `${p.degradation[k].toFixed(3)} s` }),
      el("td", { text: `${p.tyre_life[k]} laps` })));
    const table = el("table", { class: "facts" },
      el("thead", {}, el("tr", {}, el("th", { text: "Tyre" }), el("th", { text: "Pace vs medium" }),
        el("th", { text: "Wear per lap" }), el("th", { text: "Longest stint" }))),
      el("tbody", {}, ...rows));
    const sources = c.seasons ? `Measured from ${c.races} races (${c.seasons.join(", ")}).` : "No races here in the data: all-circuit averages.";
    return el("div", { class: "factbox" },
      el("dl", { class: "metrics" },
        el("div", {}, el("dt", { text: "Pit loss" }), el("dd", { text: `${Number(p.pit_loss).toFixed(1)} s` })),
        el("div", {}, el("dt", { text: "Safety car or VSC" }), el("dd", { text: pct(p.p_caution) })),
        el("div", {}, el("dt", { text: "Full safety car" }), el("dd", { text: pct(p.p_safety_car) })),
        c.track_temp != null ? el("div", {}, el("dt", { text: "Typical track temp" }), el("dd", { text: `${c.track_temp} °C` })) : null),
      table, el("p", { class: "sub", text: sources }));
  }

  function renderPlan(container, p, laps, opts = {}) {
    const result = Engine.racePlan(p, laps, opts);
    if (!result.best) {
      container.replaceChildren(el("p", { class: "warn", text: "No legal strategy with these tyres and limits. Allow more stops or another compound." }));
      return result;
    }
    const best = result.best;
    const stops = best.stops === 1 ? "1 stop" : `${best.stops} stops`;
    const windows = best.windows.map(([lo, hi], i) => `stop ${i + 1}: laps ${lo}–${hi}`).join(", ");
    const head = el("div", { class: "verdict" },
      el("p", { class: "kicker", text: "Fastest strategy" }),
      el("p", { class: "big", text: `${stops}: ${tyreWords(best.stints)}` }),
      stintBar(best.stints, laps, { windows: best.windows }),
      el("p", { class: "sub", text: best.stops ? `Box on lap ${best.pit_laps.join(" and ")}. Within 2 s of the best: ${windows}.` : "No stop." }));

    const rows = result.by_stops.map((o) => el("tr", { class: o === best ? "sel" : null },
      el("th", { scope: "row", text: o.stops === 1 ? "1 stop" : `${o.stops} stops` }),
      el("td", { class: "barcell" }, stintBar(o.stints, laps, { windows: o.windows })),
      el("td", { text: o.pit_laps.join(", ") || "–" }),
      el("td", { class: o.delta === 0 ? "best" : null, text: o.delta === 0 ? "fastest" : seconds(o.delta) })));
    const byStops = el("table", { class: "plans" },
      el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: "Best plan" }),
        el("th", { text: "Pit laps" }), el("th", { text: "vs fastest" }))),
      el("tbody", {}, ...rows));

    const alts = el("ol", { class: "alts" }, ...result.top.slice(1, 6).map((o) =>
      el("li", {}, el("span", { class: "alt-label", text: tyreWords(o.stints) }),
        el("span", { class: "alt-delta", text: seconds(o.delta) }))));

    const s = result.sensitivity;
    const same = (x) => x && x.label === best.label;
    const sens = el("p", { class: "sub" },
      same(s.degradation_low) && same(s.degradation_high)
        ? "Robust: the same plan wins if tyres wear 25% less or 25% more."
        : `If tyres wear 25% less: ${s.degradation_low?.label ?? "–"}. If 25% more: ${s.degradation_high?.label ?? "–"}. (S = soft, M = medium, H = hard)`);

    const sc = el("p", { class: "sub", text:
      `A safety car or VSC comes out in about ${pct(p.p_caution)} of races here. If one appears inside a pit window, a stop costs roughly half, so box.` });

    container.replaceChildren(head, el("h3", { text: "Best plan for each number of stops" }), byStops,
      el("h3", { text: "Next best alternatives" }), alts, sens, sc,
      el("h3", { text: "What the engine knows about this circuit" }), facts(p, p.circuit));
    return result;
  }

  // ---- planner tab ------------------------------------------------------------
  function read() {
    const circuit = $("p-circuit").value;
    const wear = Number($("p-wear").value) / 100;
    $("p-wear-out").textContent = `${Math.round(wear * 100)}%`;
    const compounds = [...document.querySelectorAll("input[name=p-comp]:checked")].map((i) => i.value);
    return {
      circuit, wear, compounds,
      laps: Number($("p-laps").value),
      pitLoss: Number($("p-pit").value) || null,
      maxStops: Number($("p-stops").value),
      rule: $("p-rule").checked,
    };
  }

  function update() {
    const f = read();
    if (!f.circuit || !(f.laps >= 5)) return;
    const p = params(f.circuit, f);
    renderPlan($("plan-out"), p, f.laps, { compounds: f.compounds, twoCompoundRule: f.rule, maxStops: f.maxStops });
    renderHistory($("plan-history"), App.data.history?.[f.circuit]);
  }

  function setCircuit(circuit) {
    fillCircuitSelect($("p-circuit"), circuit);
    const p = Engine.circuitParams(App.data.params, $("p-circuit").value);
    $("p-laps").value = p.race_laps || 57;
    $("p-pit").value = Number(p.pit_loss).toFixed(1);
    $("p-wear").value = 100;
    update();
  }

  function init() {
    setCircuit(App.data.calendar?.next?.circuit);
    $("p-circuit").addEventListener("change", () => setCircuit($("p-circuit").value));
    $("plan-form").addEventListener("input", (e) => { if (e.target.id !== "p-circuit") update(); });
    $("plan-form").addEventListener("submit", (e) => e.preventDefault());
  }

  // ---- next race ----------------------------------------------------------------
  function initNext() {
    const next = App.data.calendar?.next;
    if (!next) {
      $("next-h").textContent = "Season over";
      $("next-meta").textContent = "No race is scheduled yet. Use the planner to explore any circuit.";
      return;
    }
    const p = params(next.circuit);
    const laps = p.race_laps || 57;
    $("next-h").textContent = `${next.circuit} · round ${next.round}`;
    $("next-meta").textContent = `${formatDate(next.date)} · ${laps} laps · prediction from the strategy engine, refreshed every Monday.` +
      (p.known ? "" : " No previous race here, so it uses all-circuit averages.");
    renderPlan($("next-plan"), p, laps);
    renderHistory($("next-history"), App.data.history?.[next.circuit]);
    $("next-open").addEventListener("click", () => setCircuit(next.circuit));
  }

  return { init, initNext, setCircuit, params, renderPlan };
})();
