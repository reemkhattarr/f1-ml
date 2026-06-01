# Project 1.2 — Race-Strategy Simulator: Design

A **free-air** race-strategy simulator: estimate a driver's total race time for any
strategy (which laps to pit, which compounds), so we can find the **time-optimal**
strategy and compare it to what the team actually did and to Project 1's imitation
model.

This is the *counterfactual / optimal* angle that Project 1's imitation model
cannot provide (it predicts what teams *did*, not what was *best*). The tyre model
is **fitted from real data**, not assumed.

## The physics we model (and the one simplification that makes it tractable)

A lap time decomposes roughly as:

```
lap_time ≈ base_pace(circuit, car) + fuel_effect(fuel_load) + tyre_penalty(compound, age)
```

**Fuel cancels.** Total race distance is fixed, so the fuel load at lap *N* is the
same for every strategy. When we *compare* strategies, the fuel term and the
base-pace term cancel — they are identical across strategies. What's left, and the
entire basis of the optimization, is:

- **Tyre degradation** — different compounds degrade at different rates, so a
  strategy's cumulative tyre penalty depends on which tyres run for how long.
- **Pit-loss** — every stop costs the pit-lane time loss.

The optimizer trades these off: more stops = more pit-loss but fresher (faster)
tyres; fewer stops = less pit-loss but more degradation.

## Components

1. **Tyre degradation model** (`degradation.py`) — fuel-corrected lap time as a
   function of tyre age, fitted per `(circuit, compound)` from clean stint laps
   (degree-2 polynomial to capture the late-stint "cliff"), with a per-compound
   global fallback where a circuit×compound has too little data. Validated against
   domain truths: softer compounds are faster when fresh but degrade faster.
2. **Pit-loss model** (`pitloss.py`) — seconds lost per stop, estimated per circuit
   from the data as the extra time over the in-lap + out-lap versus two normal laps.
3. **Race simulator** (`race.py`) — given a stint plan, sum the predicted tyre
   penalties lap-by-lap plus pit-loss per stop → a comparable race "time cost".
4. **Strategy optimizer** (`optimize.py`) — enumerate plausible strategies (stop
   counts, pit laps, compounds) honouring the mandatory two-compound rule, and pick
   the minimum-cost one.
5. **Evaluation** (`evaluate.py`) — optimal strategy vs the team's actual strategy
   vs Project 1's recommendation, on real races.

## Honest scope and limitations

- **Free-air only.** No traffic, no overtaking, no dirty-air pace loss, no
  undercut/overcut track-position dynamics. The simulator answers "what stint plan
  minimises tyre + pit time in clean air", not "what wins the race on track".
- **No safety cars** in v1 (a safety car makes a stop cheap and shifts the optimum;
  deferred).
- The tyre model is **empirical** — it captures degradation observed in real
  stints, conditioned on the factors public data exposes; it is not a physical
  thermal/wear tyre model (that needs confidential Pirelli/team data).
- Degradation is fitted on the **actual stint lengths teams ran**, so extrapolating
  a compound far beyond the ages teams used it is less reliable (flagged).

## What it reuses from Project 1

The built dataset (74k stint laps), the clean-lap definition, `FUEL_S_PER_LAP`,
`SLICK_COMPOUNDS`, and the uv workspace / tooling. New code lives in
`src/tire_strategy/simulator/`.
