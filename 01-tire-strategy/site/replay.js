// Race replay: the model's pit calls against what each team did, lap by lap.
"use strict";

const Replay = (() => {
  const state = { races: [], race: null, lap: 1, driver: null, timer: null, col: {}, byDriver: {} };

  function fillSeasons() {
    const years = [...new Set(state.races.map((r) => r.year))].sort((a, b) => b - a);
    $("season").replaceChildren(...years.map((y) => el("option", { value: y, text: y })));
  }

  function fillRaces() {
    const year = Number($("season").value);
    const list = state.races.filter((r) => r.year === year);
    $("race").replaceChildren(...list.map((r) => el("option", { value: r.file, text: `Round ${r.round} · ${r.circuit}` })));
  }

  function show(race) {
    stop();
    state.race = race;
    state.col = Object.fromEntries(race.columns.map((c, i) => [c, i]));
    state.byDriver = Object.fromEntries(race.drivers.map((d) => [d.code, new Map(d.laps.map((l) => [l[state.col.lap], l]))]));
    $("lap").max = race.laps;
    // Open just before the first stop, where the calls get interesting.
    const firstStop = Math.min(...race.drivers.flatMap((d) => d.laps.filter((l) => l[state.col.team_pit]).map((l) => l[state.col.lap])));
    setLap(Number.isFinite(firstStop) ? Math.max(1, firstStop - 2) : 1);
    const keep = race.drivers.some((d) => d.code === state.driver);
    selectDriver(keep ? state.driver : race.drivers[0]?.code);
    $("rep-status").textContent = `${race.circuit} ${race.year}: ${race.drivers.length} cars, ${race.laps} laps.`;
  }

  async function load(file) {
    $("rep-status").textContent = "Loading race…";
    show(await getJSON(file));
    const r = state.race;
    history.replaceState(null, "", `?race=${r.year}-${r.round}#replay`);
  }

  function setLap(n) {
    state.lap = Math.max(1, Math.min(n, state.race.laps));
    $("lap").value = state.lap;
    $("lapnum").textContent = state.lap;
    renderTower();
    renderChart();
  }

  function callCell(pit, code, cls, label) {
    if (!pit) return el("td", {}, el("span", { class: "call none", text: "–" }));
    return el("td", {}, el("span", { class: `call ${cls}` }, `${label} `, chip(code)));
  }

  function renderTower() {
    const { race, col, lap } = state;
    const rows = [];
    for (const d of race.drivers) {
      const l = state.byDriver[d.code].get(lap);
      if (!l) continue;
      const p = l[col.pit_prob];
      const tr = el("tr", { tabindex: "0", "data-code": d.code, "aria-selected": String(d.code === state.driver), class: d.code === state.driver ? "sel" : null });
      const age = l[col.tyre_age];
      tr.append(
        el("td", { class: "car" }, el("b", { text: d.code }),
          l[col.rain] ? el("span", { class: "flag rain", text: "Rain" }) : null,
          l[col.caution] ? el("span", { class: "flag", text: "SC/VSC" }) : null,
          el("small", { text: d.team })),
        el("td", { class: "tyre" }, chip(l[col.compound]), el("span", { class: "age", text: age == null ? "" : `${age} ${age === 1 ? "lap" : "laps"}` })),
        el("td", {},
          el("div", { class: "bar", role: "img", "aria-label": `${Math.round(p * 100)} percent` },
            el("i", { style: `width:${(p * 100).toFixed(1)}%` }), el("b", { style: `left:${(race.threshold * 100).toFixed(1)}%` })),
          el("span", { class: "pct", text: `${Math.round(p * 100)}%` })),
        callCell(l[col.model_pit], l[col.model_compound], "box", "Box →"),
        callCell(l[col.team_pit], l[col.team_compound], "pit", "Pitted →"));
      rows.push(tr);
    }
    $("tower").tBodies[0].replaceChildren(...rows);
  }

  function selectDriver(code) {
    state.driver = code;
    for (const tr of $("tower").tBodies[0].rows) {
      const on = tr.dataset.code === code;
      tr.classList.toggle("sel", on);
      tr.setAttribute("aria-selected", String(on));
    }
    renderChart();
    renderPlanComparison();
  }

  function actualStints(d) {
    const { col } = state;
    const stints = [];
    for (const l of d.laps) {
      const c = l[col.compound] || "?";
      const last = stints[stints.length - 1];
      const newStint = !last || last.compound !== c || (last.pitBefore);
      if (newStint) stints.push({ compound: c, laps: 1, pitBefore: false });
      else last.laps += 1;
      if (l[col.team_pit]) stints[stints.length - 1].pitBefore = true;
    }
    return stints;
  }

  function renderPlanComparison() {
    const box = $("drv-plan");
    const d = state.race?.drivers.find((x) => x.code === state.driver);
    if (!d || !App.data.params) return box.replaceChildren();
    const race = state.race;
    const p = Engine.circuitParams(App.data.params, race.circuit);
    const plan = Engine.racePlan(p, race.laps).best;
    const did = actualStints(d);
    const wet = did.some((s) => s.compound === "I" || s.compound === "W");
    box.replaceChildren(
      el("h4", { text: "Strategy: what they did vs the engine's fastest plan" }),
      el("div", { class: "compare" },
        el("span", { text: d.code }), stintBar(did, race.laps),
        el("span", { text: "Engine" }), plan ? stintBar(plan.stints, race.laps, { windows: plan.windows }) : el("span", { text: "–" })),
      el("p", { class: "sub", text: wet
        ? "Wet race: the engine plans dry strategies only, so compare with care."
        : "The engine knows tyre wear and pit loss but not traffic, safety cars that happened, or the team's race situation." }));
  }

  function renderChart() {
    const svg = $("chart");
    const { race, col } = state;
    const d = race?.drivers.find((x) => x.code === state.driver);
    if (!d) return svg.replaceChildren();
    const W = 800, H = 220, P = 24;
    const x = (lap) => P + ((lap - 1) / Math.max(1, race.laps - 1)) * (W - 2 * P);
    const y = (p) => H - 30 - p * (H - 50);
    const ns = "http://www.w3.org/2000/svg";
    const s = (tag, attrs) => {
      const n = document.createElementNS(ns, tag);
      for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
      return n;
    };
    const parts = [];
    for (const l of d.laps) {
      parts.push(s("rect", { x: x(l[col.lap]) - (W - 2 * P) / race.laps / 2, y: H - 18, width: (W - 2 * P) / race.laps + 0.5, height: 10, class: `band c-${l[col.compound]}`, fill: "currentColor" }));
    }
    parts.push(s("line", { x1: P, x2: W - P, y1: y(race.threshold), y2: y(race.threshold), stroke: "#15233b", "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
    for (const l of d.laps) {
      if (l[col.team_pit]) parts.push(s("line", { x1: x(l[col.lap]), x2: x(l[col.lap]), y1: y(1), y2: y(0), stroke: "#c8102e", "stroke-width": 3 }));
    }
    const pts = d.laps.map((l) => `${x(l[col.lap]).toFixed(1)},${y(l[col.pit_prob]).toFixed(1)}`).join(" ");
    parts.push(s("polyline", { points: pts, fill: "none", stroke: "#6a2bd9", "stroke-width": 2.5, "stroke-linejoin": "round" }));
    parts.push(s("line", { x1: x(state.lap), x2: x(state.lap), y1: y(1), y2: H - 8, stroke: "#56627a", "stroke-width": 1 }));
    svg.replaceChildren(...parts);
    const stops = d.laps.filter((l) => l[col.team_pit]).map((l) => `lap ${l[col.lap]} (${NAMES[l[col.team_compound]] || "?"})`);
    $("drv-h").textContent = `${d.code} · ${d.team}${stops.length ? ` · pitted ${stops.join(", ")}` : " · no stops"}`;
  }

  function stop() {
    clearInterval(state.timer);
    state.timer = null;
    $("play").textContent = "Play";
    $("play").setAttribute("aria-pressed", "false");
  }

  function togglePlay() {
    if (state.timer) return stop();
    if (state.lap >= state.race.laps) setLap(1);
    $("play").textContent = "Pause";
    $("play").setAttribute("aria-pressed", "true");
    state.timer = setInterval(() => (state.lap >= state.race.laps ? stop() : setLap(state.lap + 1)), 700);
  }

  async function loadLive(e) {
    e.preventDefault();
    const year = $("live-year").value, rnd = $("live-round").value;
    $("rep-status").textContent = `Loading ${year} round ${rnd} live (can take a minute the first time)…`;
    try {
      show(await getJSON(`api/replay?year=${year}&round=${rnd}`));
    } catch (err) {
      $("rep-status").textContent = `Could not load that race (${err.message}).`;
    }
  }

  async function init(races) {
    state.races = races;
    fillSeasons();
    const wanted = new URLSearchParams(location.search).get("race");
    const pick = races.find((r) => `${r.year}-${r.round}` === wanted) || races[races.length - 1];
    $("season").value = pick.year;
    fillRaces();
    $("race").value = pick.file;
    $("season").addEventListener("change", () => { fillRaces(); load($("race").value); });
    $("race").addEventListener("change", (e) => load(e.target.value));
    $("lap").addEventListener("input", (e) => { stop(); setLap(Number(e.target.value)); });
    $("play").addEventListener("click", togglePlay);
    const pickRow = (e) => { const tr = e.target.closest("tr[data-code]"); if (tr) selectDriver(tr.dataset.code); };
    $("tower").addEventListener("click", pickRow);
    $("tower").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickRow(e); } });
    $("live-form").addEventListener("submit", loadLive);
    show(await getJSON(pick.file));
  }

  return { init, stop };
})();
