# Keeping the data fresh

F1 produces new data every race weekend. This project is built so a refresh is a
single command, and it can be **automated** three ways depending on how much
infrastructure you want. The default is manual; the automation paths are wired up
and documented below so the workflow is reproducible end to end.

## What a refresh does

| Step | Command | Effect |
|---|---|---|
| 1. Add new races to the model dataset | `just update-data 2025 2026` | Incrementally appends any missing race to `tire_strategy_dataset.parquet` (FastF1-cached, idempotent). |
| 2. Rebuild model bundle | `just refresh` | = `update-data` + retrain both heads + assemble the demo Space bundle. |
| 3. Rebuild explorer data | `just refresh-explorer` | Re-extracts **all sessions** (FP/Q/Sprint/R) for every season into the explorer parquets + assembles the explorer Space bundle. |
| 4. Both at once | `just refresh-all` | Steps 2 + 3. |
| 5. Deploy | `just deploy-all` | Pushes both Hugging Face Spaces live. |

Everything keys off the FastF1 on-disk cache (`data/cache/`), so re-runs only
download what is genuinely new — a post-weekend refresh is fast, a cold rebuild is
slow but resumable (an interrupted run continues from the cache).

```bash
# After a race weekend, from the repo root:
just refresh-all      # update data, retrain, rebuild both Space bundles
just deploy-all       # push both Spaces live  (needs `hf auth login` once)
```

## Option A — Manual (default)

Run the two commands above when you want to update. No infrastructure, full
control. This is the recommended default for a personal portfolio.

## Option B — Scheduled on your machine (cron / launchd)

Run the refresh automatically on a timer on a machine that is regularly on.

**cron** (Linux, or macOS) — every Monday at 09:00, the day after most races:

```cron
# crontab -e
0 9 * * 1  cd /path/to/f1-ml && /usr/bin/env just refresh-all && just deploy-all >> /tmp/f1-refresh.log 2>&1
```

**launchd** (the native macOS scheduler) — drop this at
`~/Library/LaunchAgents/com.f1ml.refresh.plist` and `launchctl load` it:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.f1ml.refresh</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string><string>-lc</string>
    <string>cd /path/to/f1-ml && just refresh-all && just deploy-all</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>9</integer></dict>
  <key>StandardOutPath</key><string>/tmp/f1-refresh.log</string>
  <key>StandardErrorPath</key><string>/tmp/f1-refresh.err</string>
</dict></plist>
```

## Option C — GitHub Actions (cloud, always-on)

The most "production" option: a scheduled CI job refreshes and redeploys with no
machine of yours running. The workflow lives at
[`.github/workflows/refresh-data.yml`](../../.github/workflows/refresh-data.yml).
It is **opt-in** — it only runs on a manual trigger until you uncomment the
`schedule:` block — and needs one repository secret:

- `HF_TOKEN` — a Hugging Face write token (Settings → Access Tokens), added under
  the repo's *Settings → Secrets and variables → Actions*.

Enable it by setting that secret and uncommenting the weekly `schedule:` trigger.
Trigger a one-off run anytime from the Actions tab ("Run workflow").

> Note: a cold all-sessions rebuild is heavy for a free CI runner. In practice the
> scheduled job should run the **incremental** model refresh (fast); a full
> explorer rebuild is best run locally and committed, or pointed at a cache.
