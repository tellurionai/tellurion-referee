// App shell: scenario picker, #scenario=<id> routing, wiring, technical details.

import "./styles.css";
import { Meters } from "./meters";
import {
  PLAIN_VERDICT,
  type Replay,
  type ReplayIndex,
  ReplayError,
  type ScenarioEntry,
  flagFrame,
  loadIndex,
  loadReplay,
  sample,
} from "./replay";
import { Stage } from "./stage";
import { Timeline } from "./timeline";
import { VerdictCard } from "./verdict";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const stage = new Stage($<HTMLCanvasElement>("stage"));
const meters = new Meters($("meter-ai"), $("meter-ref"));
const card = new VerdictCard($("verdict"));
const techList = $("tech-list");
let current: Replay | null = null;
let entry: ScenarioEntry | null = null;
let index: ReplayIndex;
const cache = new Map<string, Replay>();

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
  if (!current) return;
  stage.show(pos);
  meters.show(pos);
  card.show(pos);
  renderTech(pos);
}

function fmt(v: number | null, digits = 3) {
  return v === null ? "no reading" : v.toFixed(digits);
}

function renderTech(pos: number) {
  const r = current!, f = r.frames;
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
    ["Selection rule", entry?.selection_rule ?? "–"],
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

function renderPicker(activeId: string) {
  const nav = $("picker");
  nav.replaceChildren(
    ...index.scenarios.map((s) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = s.title;
      b.setAttribute("aria-current", String(s.id === activeId));
      b.addEventListener("click", () => (location.hash = `scenario=${s.id}`));
      return b;
    }),
  );
  // Keep the active chip visible in the horizontal strip without scrolling the page.
  const chip = nav.querySelector<HTMLElement>('[aria-current="true"]');
  if (chip && (chip.offsetLeft < nav.scrollLeft || chip.offsetLeft + chip.offsetWidth > nav.scrollLeft + nav.clientWidth)) {
    nav.scrollLeft = chip.offsetLeft - 16;
  }
}

const autoplay = () => {
  if (!matchMedia("(prefers-reduced-motion: reduce)").matches) timeline.play();
};

function idFromHash(): string {
  const m = location.hash.match(/scenario=([\w-]+)/);
  const id = m?.[1];
  return index.scenarios.some((s) => s.id === id) ? id! : index.scenarios[0].id;
}

async function open(id: string) {
  const e = index.scenarios.find((s) => s.id === id)!;
  renderPicker(id);
  let r = cache.get(id);
  if (!r) {
    r = await loadReplay(e);
    cache.set(id, r);
  }
  entry = e;
  current = r;
  $("scenario-title").textContent = r.title ?? e.title;
  $("scenario-description").textContent = r.description ?? "";
  stage.load(r);
  meters.load(r);
  card.load(r);
  timeline.load({
    frames: r.frames.t.length,
    dt: r.dt,
    onsetFrame: r.onset_step,
    flagFrame: flagFrame(r),
    onFrame: render,
  });
}

function showError(err: unknown) {
  const main = document.querySelector("main")!;
  const div = document.createElement("div");
  div.className = "error";
  const problems = err instanceof ReplayError ? err.problems : [String(err)];
  div.innerHTML = "<strong>Couldn't load replay data.</strong>";
  const ul = document.createElement("ul");
  for (const p of problems) {
    const li = document.createElement("li");
    li.textContent = err instanceof ReplayError ? `${err.source}: ${p}` : p;
    ul.append(li);
  }
  div.append(ul);
  main.replaceChildren(div);
}

async function boot() {
  index = await loadIndex();
  $("demo-banner").hidden = !index.is_demo_data;
  $("provenance").textContent = `Replays: ${index.generated_from}`;
  window.addEventListener("hashchange", () => open(idFromHash()).then(autoplay).catch(showError));
  await open(idFromHash());
  autoplay();
}

boot().catch(showError);
