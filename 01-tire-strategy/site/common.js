// Shared helpers for the tyre-strategy pages.
"use strict";

const NAMES = { S: "Soft", M: "Medium", H: "Hard", I: "Intermediate", W: "Wet" };
const LONG = { SOFT: "S", MEDIUM: "M", HARD: "H", INTERMEDIATE: "I", WET: "W" };
const $ = (id) => document.getElementById(id);
const App = { data: {}, local: false };

async function getJSON(path, optional = false) {
  const res = await fetch(path);
  if (!res.ok) {
    if (optional) return null;
    throw new Error(`${path}: ${res.status}`);
  }
  return res.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null && c !== false) node.append(c);
  return node;
}

function chip(code, title) {
  return el("span", { class: `chip c-${code || "x"}`, title: title || NAMES[code] || "Unknown", text: code || "?" });
}

const seconds = (v) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} s`);
const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const tyreWords = (stints) => stints.map((s) => NAMES[LONG[s.compound] || s.compound] || s.compound).join(" → ");
const formatDate = (iso) => new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });

// A strategy drawn as stints across the race distance, with optional pit windows.
// `stints`: [{compound, laps}] (compound long name or letter); `from` = first lap shown.
function stintBar(stints, totalLaps, { windows = [], from = 1, now = null } = {}) {
  const span = totalLaps - from + 1;
  const bar = el("div", { class: "stintbar", role: "img",
    "aria-label": stints.map((s) => `${NAMES[LONG[s.compound] || s.compound]} ${s.laps} laps`).join(", ") });
  let lap = from;
  for (const s of stints) {
    const code = LONG[s.compound] || s.compound;
    const seg = el("span", { class: `seg c-${code}`, style: `width:${(100 * s.laps) / span}%`,
      title: `${NAMES[code] || code}: laps ${lap}–${lap + s.laps - 1}` },
      el("b", { text: code }), s.laps >= 4 ? el("small", { text: `${s.laps}` }) : null);
    bar.append(seg);
    lap += s.laps;
  }
  for (const [lo, hi] of windows) {
    bar.append(el("i", { class: "window", title: `Pit window: laps ${lo}–${hi}`,
      style: `left:${(100 * (lo - from)) / span}%;width:${(100 * (hi - lo + 1)) / span}%` }));
  }
  if (now != null) bar.append(el("i", { class: "now", style: `left:${(100 * (now - from)) / span}%` }));
  return bar;
}

function renderHistory(container, races) {
  if (!races || !races.length) {
    container.replaceChildren(el("p", { class: "sub", text: "No recent races here in the data." }));
    return;
  }
  const blocks = races.slice(0, 2).map((race) => {
    const rows = race.drivers.map((d) =>
      el("li", {}, el("span", { class: "pos", text: d.pos }), el("b", { text: d.code }),
        stintBar(d.stints.map(([c, n]) => ({ compound: c, laps: n })), race.laps)));
    return el("div", { class: "race-history" },
      el("h4", { text: `${race.year}${race.wet ? " · wet race" : ""}` }), el("ol", {}, ...rows));
  });
  container.replaceChildren(...blocks);
}

function circuitNames() {
  return Object.keys(App.data.params?.circuits || {}).sort((a, b) => a.localeCompare(b));
}

function fillCircuitSelect(select, selected) {
  select.replaceChildren(...circuitNames().map((c) => el("option", { value: c, text: c })));
  if (selected && circuitNames().includes(selected)) select.value = selected;
}
