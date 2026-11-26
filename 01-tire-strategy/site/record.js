// Track record: how the engine's plans compare with what the top ten really did.
"use strict";

const Record = (() => {
  const share = (v) => (v == null ? "–" : pct(v));
  const lap = (v) => (v == null ? "–" : `lap ${Math.round(v)}`);

  function scoreRow(label, rates) {
    return el("tr", {},
      el("th", { scope: "row", text: label }),
      el("td", { text: share(rates.stops_match) }),
      el("td", { text: share(rates.baseline_one_stop) }),
      el("td", { text: share(rates.baseline_last_time) }),
      el("td", { text: share(rates.in_window) }),
      el("td", { text: rates.first_stop_error_median == null ? "–" : `${rates.first_stop_error_median} laps` }),
      el("td", { text: rates.races }));
  }

  function verdict(s) {
    const c = s.clean || s;
    if (!c.races) return "Not enough past races to judge yet.";
    const diff = c.stops_match - c.baseline_one_stop;
    if (diff > 0.05) return `On races without a safety car deciding the stops, the engine picks the top ten's number of stops more often than "always one stop" (${share(c.stops_match)} vs ${share(c.baseline_one_stop)}).`;
    if (diff < -0.05) return `Honest result: on races without a safety car deciding the stops, the engine does not yet beat simply saying "one stop" (${share(c.stops_match)} vs ${share(c.baseline_one_stop)}). Treat its plans as a starting point, not a call.`;
    return `On races without a safety car deciding the stops, the engine is level with simply saying "one stop" (${share(c.stops_match)} vs ${share(c.baseline_one_stop)}).`;
  }

  function raceRow(r) {
    const ok = r.stops_match;
    return el("tr", {},
      el("td", { text: `${r.year} R${r.round}` }),
      el("td", { text: r.circuit }),
      el("td", {}, ...r.predicted.split("-").map((c) => chip(c, NAMES[c]))),
      el("td", { text: `${r.predicted_stops} · ${lap(r.predicted_first_stop)}` }),
      el("td", { text: `${r.actual_stops} (${share(r.actual_share)}) · ${lap(r.actual_first_stop)}` }),
      el("td", { class: ok ? "hit" : "miss", text: ok ? "Same stops" : "Different" },
        r.caution_shaped ? el("span", { class: "flag", title: "Top-ten stops were made under a safety car, VSC or red flag", text: "SC" }) : null));
  }

  function liveRow(e) {
    const res = e.result;
    const outcome = !res ? "Race not run yet"
      : res.wet ? "Wet race, not scored"
      : `${res.actual_stops} stops by the top ten (${share(res.actual_share)}), first at ${lap(res.actual_first_stop)}: ${res.stops_match ? "same stops" : "different"}`;
    return el("tr", {},
      el("td", { text: `${e.year} R${e.round}` }),
      el("td", { text: e.circuit }),
      el("td", { text: e.made_at.slice(0, 10) }),
      el("td", {}, ...e.plan.label.split("-").map((c) => chip(c, NAMES[c])), el("span", { class: "age", text: ` stop${e.plan.pit_laps.length === 1 ? "" : "s"} ${e.plan.pit_laps.join(", ") || "none"}` })),
      el("td", { text: outcome }));
  }

  function table(head, rows) {
    return el("div", { class: "scroll" }, el("table", { class: "tower record" },
      el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { scope: "col", text: h })))),
      el("tbody", {}, ...rows)));
  }

  function init(record) {
    const box = $("record");
    if (!record) {
      box.replaceChildren(el("p", { class: "status", text: "The track record appears after the next weekly update." }));
      return;
    }
    const s = record.summary;
    const live = record.live || [];
    box.replaceChildren(
      el("p", { class: "lead-sm", text: verdict(s) }),
      table(["", "Same stops as top ten", "Always one stop", "Same as last year", "Top ten's first stop in window", "First-stop gap (median)", "Races"],
        [scoreRow("Races without SC stops", s.clean || {}), scoreRow("All dry races", s)]),
      el("p", { class: "sub", text: "For each race since 2024 the engine was fitted only on the races before it, so it could not peek. \"Same stops\" means it chose the number of stops most of the top ten used. The two baselines are simple rules anyone could follow. Wet races are left out: the engine plans dry races." }),
      el("h3", { text: "Predictions made before the race" }),
      live.length
        ? table(["Race", "Circuit", "Made", "Plan", "What happened"], live.slice().reverse().map(liveRow))
        : el("p", { class: "sub", text: "Each weekly update saves the plan for the next race before it starts; it is scored here once the race has run." }),
      el("h3", { text: "Every backtested race" }),
      table(["Race", "Circuit", "Plan", "Plan stops · first", "Top ten · first", ""], s.races === 0 ? [] : record.races.slice().reverse().map(raceRow)));
  }

  return { init };
})();
