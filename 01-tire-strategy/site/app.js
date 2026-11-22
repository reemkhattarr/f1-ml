// Boot: load the exported data, wire the tabs, and switch on local-only tools.
"use strict";

const TABS = ["next", "plan", "advise", "replay", "model"];

function route() {
  const tab = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "next";
  for (const t of TABS) $(`tab-${t}`).hidden = t !== tab;
  for (const a of document.querySelectorAll(".tabs a")) {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (tab !== "replay") Replay.stop();
}

function renderModelCard(card) {
  const m = card.metrics || {};
  const fmt = (v, isPct) => (v == null ? "–" : isPct ? pct(v) : v.toFixed(3));
  const items = [
    ["Strategic cost (model)", fmt(m.cv_model_cost_mean), true],
    ["Never-pit baseline", fmt(m.cv_majority_cost_mean)],
    ["Rule-based baseline", fmt(m.cv_rule_cost_mean)],
    ["Tyre choice accuracy", fmt(m.cv_compound_accuracy_mean, true)],
  ];
  const sp = card.split || {};
  $("card").replaceChildren(
    el("dl", { class: "metrics" }, ...items.map(([k, v, best]) =>
      el("div", {}, el("dt", { text: k }), el("dd", { class: best ? "best" : null, text: v })))),
    el("p", { class: "sub", text:
      `Cost: lower is better; a missed stop counts five times a false alarm. Cross-validated over ${sp.n_races ?? "all"} races` +
      (sp.holdout_season ? `, tested on ${sp.holdout_season} races it never trained on` : "") +
      (card.trained_at ? `. Trained ${card.trained_at.slice(0, 10)}.` : ".") }));
}

function renderEngineCard(params) {
  const d = params.default;
  const circuits = Object.keys(params.circuits).length;
  $("engine-card").replaceChildren(
    el("dl", { class: "metrics" },
      el("div", {}, el("dt", { text: "Circuits measured" }), el("dd", { text: circuits })),
      el("div", {}, el("dt", { text: "Seasons" }), el("dd", { text: `${params.seasons[0]}–${params.seasons[params.seasons.length - 1]}` })),
      el("div", {}, el("dt", { text: "Typical pit loss" }), el("dd", { text: `${d.pit_loss.toFixed(1)} s` })),
      el("div", {}, el("dt", { text: "Track gets faster" }), el("dd", { text: `${Math.abs(params.track_evolution_s_per_lap).toFixed(3)} s/lap` }))));
}

async function detectLocal() {
  try {
    const res = await fetch("api/health", { cache: "no-store" });
    if (!res.ok) return false;
    const body = await res.json();
    return body.status === "ok";
  } catch {
    return false;
  }
}

async function refresh() {
  const btn = $("refresh");
  btn.disabled = true;
  $("refresh-status").textContent = "Fetching new races and refitting… this can take several minutes.";
  try {
    const res = await fetch("api/refresh", { method: "POST" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.status);
    $("refresh-status").textContent = `${body.message} Reloading…`;
    setTimeout(() => location.reload(), 1500);
  } catch (err) {
    $("refresh-status").textContent = `Refresh failed: ${err.message}`;
    btn.disabled = false;
  }
}

async function main() {
  window.addEventListener("hashchange", route);
  route();
  try {
    const [races, card, params, calendar, history, models] = await Promise.all([
      getJSON("data/races.json"), getJSON("data/model.json"),
      getJSON("data/strategy_params.json"), getJSON("data/calendar.json", true),
      getJSON("data/history.json", true), getJSON("data/models.json", true),
    ]);
    Object.assign(App.data, { races, card, params, calendar, history, models });
    renderModelCard(card);
    renderEngineCard(params);
    Planner.initNext();
    Planner.init();
    Advisor.init();
    $("status").textContent = "";
    await Replay.init(races);
  } catch (err) {
    $("status").textContent = `Could not load the data (${err.message}).`;
  }
  App.local = await detectLocal();
  if (App.local) {
    $("mode").hidden = false;
    $("live-form").hidden = false;
    $("local-tools").hidden = false;
    $("refresh").addEventListener("click", refresh);
  }
}

main();
