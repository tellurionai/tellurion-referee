// App shell: scenario picker, #scenario=<id> routing, wiring, technical details.

import "./styles.css";
import { App, idFromHash, problemsOf } from "./app";
import { Meters } from "./meters";
import { PLAIN_VERDICT, type ReplayIndex, flagFrame, loadIndex, loadReplay, sample } from "./replay";
import { Stage } from "./stage";
import { Timeline } from "./timeline";
import { VerdictCard } from "./verdict";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const stage = new Stage($<HTMLCanvasElement>("stage"));
const meters = new Meters($("meter-ai"), $("meter-ref"));
const card = new VerdictCard($("verdict"));
const techList = $("tech-list");
const errorPanel = $("load-error");
let app: App | null = null;

const timeline = new Timeline(
  $<HTMLInputElement>("scrub"),
  $<HTMLButtonElement>("play"),
  document.getElementById("play-icon") as unknown as SVGPathElement,
  $("clock"),
  { onset: $("mark-onset"), flag: $("mark-flag"), onsetLabel: $("label-onset"), flagLabel: $("label-flag") },
);

const slowmo = $<HTMLInputElement>("slowmo");
slowmo.addEventListener("change", () => (timeline.slowmo = slowmo.checked));

function render(pos: number) {
  if (!app?.current) return;
  stage.show(pos);
  meters.show(pos);
  card.show(pos);
  renderTech(pos);
}

function fmt(v: number | null, digits = 3) {
  return v === null ? "no reading" : v.toFixed(digits);
}

function renderTech(pos: number) {
  const r = app!.current!, f = r.frames;
  const k = Math.floor(pos);
  const meas = sample(f.measured_theta, pos), pred = sample(f.pred_theta, pos);
  const rows: [string, string][] = [
    ["Frame", `${k} / ${f.t.length - 1}`],
    ["θ reality (rad)", fmt(sample(f.true_theta, pos))],
    ["θ sensor (rad)", fmt(meas)],
    ["θ AI (ensemble mean, rad)", fmt(pred)],
    ["Residual |sensor − AI|", meas === null || pred === null ? "–" : Math.abs(meas - pred).toFixed(3)],
    ["AI confidence (ensemble agreement)", fmt(sample(f.ai_confidence, pos), 2)],
    ["Referee score", fmt(sample(f.referee_score, pos), 2)],
    ["Referee verdict", f.verdict[k]],
    ["Hidden switch", r.perturbation.type === "none" ? "none" : `${r.perturbation.type} / ${r.perturbation.subtype} / ${r.perturbation.severity}`],
    ["Switch flips at", r.onset_step < 0 ? "–" : `frame ${r.onset_step} (${(r.onset_step * r.dt).toFixed(2)} s)`],
    ["Answer key", `${r.expected_verdict} (${PLAIN_VERDICT[r.expected_verdict]})`],
    ["Selection rule", app!.entry?.selection_rule ?? "–"],
  ];
  techList.replaceChildren(
    ...rows.flatMap(([a, b]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = a;
      dd.textContent = b;
      return [dt, dd];
    }),
  );
}

function renderPicker(index: ReplayIndex) {
  $("picker").replaceChildren(
    ...index.scenarios.map((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.dataset.id = s.id;
      b.textContent = s.title;
      b.addEventListener("click", () => (location.hash = `scenario=${s.id}`));
      return b;
    }),
  );
}

function highlight(id: string) {
  const nav = $("picker");
  for (const b of nav.querySelectorAll<HTMLElement>(".chip")) b.setAttribute("aria-current", String(b.dataset.id === id));
  // Keep the active chip visible in the horizontal strip without scrolling the page.
  const chip = nav.querySelector<HTMLElement>('[aria-current="true"]');
  if (chip && (chip.offsetLeft < nav.scrollLeft || chip.offsetLeft + chip.offsetWidth > nav.scrollLeft + nav.clientWidth)) {
    nav.scrollLeft = chip.offsetLeft - 16;
  }
}

// Errors go in their own panel; the player stays in place so the next selection can recover.
function showError(problems: string[]) {
  const ul = document.createElement("ul");
  for (const p of problems) {
    const li = document.createElement("li");
    li.textContent = p;
    ul.append(li);
  }
  const title = document.createElement("strong");
  title.textContent = "Couldn't load replay data.";
  errorPanel.replaceChildren(title, ul);
  errorPanel.hidden = false;
}

function clearError() {
  errorPanel.hidden = true;
  errorPanel.replaceChildren();
}

const autoplay = (shown: boolean) => {
  if (shown && !matchMedia("(prefers-reduced-motion: reduce)").matches) timeline.play();
};

async function boot() {
  let index: ReplayIndex;
  try {
    index = await loadIndex();
  } catch (err) {
    showError(problemsOf(err));
    return;
  }
  $("demo-banner").hidden = !index.is_demo_data;
  $("provenance").textContent = `Replays: ${index.generated_from}`;
  renderPicker(index);
  app = new App(index, loadReplay, {
    highlight,
    showError,
    clearError,
    pause: () => timeline.pause(),
    show(entry, r) {
      $("scenario-title").textContent = r.title ?? entry.title;
      $("scenario-description").textContent = r.description ?? "";
      stage.load(r);
      meters.load(r);
      card.load(r);
      timeline.load({ frames: r.frames.t.length, dt: r.dt, onsetFrame: r.onset_step, flagFrame: flagFrame(r), onFrame: render });
    },
  });
  window.addEventListener("hashchange", () => app!.open(idFromHash(location.hash, index)).then(autoplay));
  autoplay(await app.open(idFromHash(location.hash, index)));
}

boot();
