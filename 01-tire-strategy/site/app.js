// Tyre strategy replay: static page reading JSON exported by tire_strategy.export_demo.
"use strict";

const NAMES = { S: "Soft", M: "Medium", H: "Hard", I: "Intermediate", W: "Wet" };
const $ = (id) => document.getElementById(id);
const state = { races: [], race: null, lap: 1, driver: null, timer: null, col: {} };

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

function chip(code, title) {
  return el("span", { class: `chip c-${code || "x"}`, title: title || NAMES[code] || "Unknown", text: code || "?" });
}

// ---- race selection ---------------------------------------------------------
function fillSeasons() {
  const years = [...new Set(state.races.map((r) => r.year))].sort((a, b) => b - a);
  $("season").replaceChildren(...years.map((y) => el("option", { value: y, text: y })));
}

function fillRaces() {
  const year = Number($("season").value);
  const list = state.races.filter((r) => r.year === year);
  $("race").replaceChildren(
    ...list.map((r) => el("option", { value: r.file, text: `Round ${r.round} · ${r.circuit}` }))
  );
}

async function loadRace(file) {
  stop();
  $("status").textContent = "Loading race…";
  const race = await getJSON(file);
  state.race = race;
  state.col = Object.fromEntries(race.columns.map((c, i) => [c, i]));
  state.byDriver = Object.fromEntries(
    race.drivers.map((d) => [d.code, new Map(d.laps.map((l) => [l[state.col.lap], l]))])
  );
  $("lap").max = race.laps;
  setLap(1);
  const keep = race.drivers.some((d) => d.code === state.driver);
  selectDriver(keep ? state.driver : race.drivers[0]?.code);
  $("status").textContent = `${race.circuit} ${race.year}: ${race.drivers.length} cars, ${race.laps} laps.`;
  const params = new URLSearchParams({ race: `${race.year}-${race.round}` });
  history.replaceState(null, "", `?${params}`);
}

// ---- lap view ---------------------------------------------------------------
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
    if (!l) continue; // retired or not running this lap
    const p = l[col.pit_prob];
    const tr = el("tr", { tabindex: "0", "data-code": d.code, "aria-selected": String(d.code === state.driver) });
    if (d.code === state.driver) tr.classList.add("sel");
    const flags = [];
    if (l[col.rain]) flags.push(el("span", { class: "flag rain", text: "Rain" }));
    if (l[col.caution]) flags.push(el("span", { class: "flag", text: "SC/VSC" }));
    tr.append(
      el("td", { class: "car" }, el("b", { text: d.code }), ...flags, el("small", { text: d.team })),
      el("td", { class: "tyre" }, chip(l[col.compound]), el("span", { class: "age", text: l[col.tyre_age] == null ? "" : `${l[col.tyre_age]} ${l[col.tyre_age] === 1 ? "lap" : "laps"}` })),
      el(
        "td",
        {},
        el("div", { class: "bar", role: "img", "aria-label": `${Math.round(p * 100)} percent` },
          el("i", { style: `width:${(p * 100).toFixed(1)}%` }),
          el("b", { style: `left:${(race.threshold * 100).toFixed(1)}%` })),
        el("span", { class: "pct", text: `${Math.round(p * 100)}%` })
      ),
      callCell(l[col.model_pit], l[col.model_compound], "box", "Box →"),
      callCell(l[col.team_pit], l[col.team_compound], "pit", "Pitted →")
    );
    rows.push(tr);
  }
  $("tower").tBodies[0].replaceChildren(...rows);
}

// ---- driver chart -----------------------------------------------------------
function selectDriver(code) {
  state.driver = code;
  for (const tr of $("tower").tBodies[0].rows) {
    const on = tr.dataset.code === code;
    tr.classList.toggle("sel", on);
    tr.setAttribute("aria-selected", String(on));
  }
  renderChart();
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
  // compound band along the bottom
  for (const l of d.laps) {
    const code = l[col.compound];
    parts.push(s("rect", { x: x(l[col.lap]) - (W - 2 * P) / race.laps / 2, y: H - 18, width: (W - 2 * P) / race.laps + 0.5, height: 10, class: `band c-${code}`, fill: "currentColor" }));
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

// ---- playback -----------------------------------------------------------------
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

// ---- model card -----------------------------------------------------------------
function renderCard(card) {
  const m = card.metrics || {};
  const fmt = (v, pct) => (v == null ? "–" : pct ? `${Math.round(v * 100)}%` : v.toFixed(3));
  const items = [
    ["Strategic cost (model)", fmt(m.cv_model_cost_mean), true],
    ["Never-pit baseline", fmt(m.cv_majority_cost_mean)],
    ["Rule-based baseline", fmt(m.cv_rule_cost_mean)],
    ["Compound accuracy", fmt(m.cv_compound_accuracy_mean, true)],
  ];
  const dl = el("dl", { class: "metrics" },
    ...items.map(([k, v, best]) => el("div", {}, el("dt", { text: k }), el("dd", { class: best ? "best" : "", text: v }))));
  const sp = card.split || {};
  const note = el("p", { class: "pct", text:
    `Cost: lower is better; a missed pit counts five times a false alarm. Cross-validated over ${sp.n_races ?? "all"} races. ` +
    (sp.holdout_season ? `Tested on ${sp.holdout_season} races it never trained on. ` : "") +
    (card.trained_at ? `Trained ${card.trained_at.slice(0, 10)}.` : "") });
  $("card").replaceChildren(dl, note);
}

// ---- boot ---------------------------------------------------------------------------
async function main() {
  try {
    const [races, card] = await Promise.all([getJSON("data/races.json"), getJSON("data/model.json")]);
    state.races = races;
    renderCard(card);
    fillSeasons();
    const wanted = new URLSearchParams(location.search).get("race");
    const pick = races.find((r) => `${r.year}-${r.round}` === wanted) || races[races.length - 1];
    $("season").value = pick.year;
    fillRaces();
    $("race").value = pick.file;
    await loadRace(pick.file);
  } catch (err) {
    $("status").textContent = `Could not load the data (${err.message}).`;
  }
  $("season").addEventListener("change", () => { fillRaces(); loadRace($("race").value); });
  $("race").addEventListener("change", (e) => loadRace(e.target.value));
  $("lap").addEventListener("input", (e) => { stop(); setLap(Number(e.target.value)); });
  $("play").addEventListener("click", togglePlay);
  const pick = (e) => { const tr = e.target.closest("tr[data-code]"); if (tr) selectDriver(tr.dataset.code); };
  $("tower").addEventListener("click", pick);
  $("tower").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e); } });
}

main();
