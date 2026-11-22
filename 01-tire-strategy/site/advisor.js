// Pit-wall advisor: box or stay out, priced by the strategy engine, plus the model's view.
"use strict";

const Advisor = (() => {
  const num = (id) => { const v = $(id).value; return v === "" ? null : Number(v); };

  function read() {
    const used = [...document.querySelectorAll("input[name=a-used]:checked")].map((i) => i.value);
    return {
      circuit: $("a-circuit").value,
      total_laps: num("a-laps"),
      lap: num("a-lap"),
      compound: $("a-tyre").value,
      tyre_age: num("a-age") ?? 0,
      caution: $("a-caution").value,
      rain: $("a-rain").checked,
      gap_ahead: num("a-ahead"),
      gap_behind: num("a-behind"),
      track_temp: num("a-temp"),
      used,
    };
  }

  // The learned models want the same inputs as in training; unknown ones stay empty.
  function modelRow(s) {
    const dry = new Set([...s.used, s.compound].filter((c) => Engine.DRY.includes(c)));
    const caution = s.caution !== "none";
    return {
      lap_number: s.lap, laps_remaining: s.total_laps - s.lap, tyre_age: s.tyre_age,
      stint_position: s.tyre_age + 1, pit_count_so_far: s.used.filter((c) => c !== s.compound).length ? 1 : 0,
      track_temp: s.track_temp, air_temp: null, gap_ahead_s: s.gap_ahead, gap_behind_s: s.gap_behind,
      last_clean_lap_time_s: null, deg_slope_s_per_lap: null, relative_pace_s: null,
      compound: s.compound, circuit: s.circuit, team: "UNKNOWN", fresh_tyre: true,
      under_caution: caution, flag_sc: s.caution === "sc", flag_vsc: s.caution === "vsc",
      flag_yellow: false, used_second_compound: dry.size >= 2, rainfall: s.rain, sc_just_deployed: caution,
    };
  }

  function secondOpinion(s) {
    const models = App.data.models;
    if (!models) return el("p", { class: "sub", text: "The learned model is not available in this build." });
    const row = modelRow(s);
    const p = Model.pitProbability(models, row);
    const thr = models.pit.threshold;
    const probs = Model.compoundProbabilities(models, row);
    const pick = s.rain ? "INTERMEDIATE" : Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];
    const says = p >= thr;
    return el("div", { class: "opinion" },
      el("p", { class: "kicker", text: "Second opinion: what teams usually do here" }),
      el("p", {}, el("b", { text: says ? `Most teams would box within 3 laps (${pct(p)})` : `Most teams would stay out (${pct(p)} chance of a stop within 3 laps)` }),
        says ? el("span", {}, " onto ", chip(LONG[pick]), ` ${NAMES[LONG[pick]].toLowerCase()}s`) : null),
      el("div", { class: "bar wide", role: "img", "aria-label": `${pct(p)} chance of a stop` },
        el("i", { style: `width:${(p * 100).toFixed(1)}%` }), el("b", { style: `left:${(thr * 100).toFixed(1)}%` })),
      el("p", { class: "sub", text: "Learned from every team decision since 2022. Fill in gaps and track temperature for a sharper estimate; pace and wear history is not known here." }));
  }

  function render() {
    const s = read();
    const out = $("adv-out");
    if (!s.circuit || !s.total_laps || !s.lap) return;
    if (s.lap > s.total_laps) {
      out.replaceChildren(el("p", { class: "warn", text: "That lap is after the chequered flag." }));
      return;
    }
    const p = Planner.params(s.circuit);
    let a;
    try {
      a = Engine.advise(p, s);
    } catch (err) {
      out.replaceChildren(el("p", { class: "warn", text: err.message }));
      return;
    }
    const verdicts = {
      box: `Box now${a.compound ? ` for ${NAMES[LONG[a.compound]].toLowerCase()}s` : ""}`,
      stay: "Stay out",
      either: "Too close to call",
    };
    const why = a.delta == null ? (a.source === "wet rule" ? "Rain rule" : "")
      : a.decision === "box" ? `Boxing now is ${a.delta.toFixed(1)} s faster to the flag than the best stay-out plan.`
      : a.decision === "stay" ? `Staying out is ${(-a.delta).toFixed(1)} s faster to the flag than boxing now.`
      : `Box now and stay out are within ${Math.abs(a.delta).toFixed(1)} s; decide on traffic and track position.`;
    const card = el("div", { class: `verdict v-${a.decision}` },
      el("p", { class: "kicker", text: `Lap ${s.lap} of ${s.total_laps} · stop costs ${a.pit_loss_now.toFixed(1)} s now` }),
      el("p", { class: "big", text: verdicts[a.decision] }),
      el("p", { class: "sub", text: why }));

    const parts = [card];
    if (a.stay || a.box.length) {
      const all = [a.stay ? { name: "Stay out", plan: a.stay } : null,
        ...a.box.map((b) => ({ name: `Box now · ${NAMES[LONG[b.compound]]}`, plan: b }))].filter(Boolean);
      const best = Math.min(...all.map((o) => o.plan.cost));
      const rows = all.map((o) => el("tr", { class: o.plan.cost === best ? "sel" : null },
        el("th", { scope: "row", text: o.name }),
        el("td", { class: "barcell" }, stintBar(o.plan.stints, s.total_laps, { from: s.lap })),
        el("td", { text: o.plan.pit_laps.length ? o.plan.pit_laps.join(", ") : "–" }),
        el("td", { class: o.plan.cost === best ? "best" : null, text: o.plan.cost === best ? "fastest" : seconds(o.plan.cost - best) })));
      parts.push(el("h3", { text: "The rest of the race, each way" }),
        el("table", { class: "plans" },
          el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: `Laps ${s.lap}–${s.total_laps}` }),
            el("th", { text: "Later stops" }), el("th", { text: "vs fastest" }))),
          el("tbody", {}, ...rows)));
    }
    if (a.notes.length) parts.push(el("ul", { class: "notes" }, ...a.notes.map((n) => el("li", { text: n }))));
    parts.push(secondOpinion(s));
    out.replaceChildren(...parts);
  }

  function setCircuit(circuit) {
    fillCircuitSelect($("a-circuit"), circuit);
    const p = Engine.circuitParams(App.data.params, $("a-circuit").value);
    $("a-laps").value = p.race_laps || 57;
    const temp = App.data.params.circuits[$("a-circuit").value]?.track_temp;
    $("a-temp").placeholder = temp != null ? `typical ${temp}` : "optional";
    render();
  }

  function init() {
    setCircuit(App.data.calendar?.next?.circuit);
    $("a-circuit").addEventListener("change", () => setCircuit($("a-circuit").value));
    $("adv-form").addEventListener("input", (e) => { if (e.target.id !== "a-circuit") render(); });
    $("adv-form").addEventListener("submit", (e) => e.preventDefault());
  }

  return { init, setCircuit, modelRow };
})();
