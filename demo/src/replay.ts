// Loads and validates replays/index.json and replays/<id>.json (demo/REPLAY_FORMAT.md).
// The UI only ever draws a replay that passed validate().

export const VERDICTS = ["trust", "model_error", "instrument_fault", "discovery"] as const;
export type Verdict = (typeof VERDICTS)[number];

export interface ScenarioEntry {
  id: string;
  title: string;
  file: string;
  expected_verdict: Verdict;
  selection_rule: string;
  missed?: boolean;
}

export interface ReplayIndex {
  generated_from: string;
  is_demo_data: boolean;
  scenarios: ScenarioEntry[];
}

export interface Frames {
  t: number[];
  true_theta: number[];
  measured_theta: (number | null)[];
  pred_theta: (number | null)[];
  ai_confidence: (number | null)[];
  referee_score: (number | null)[];
  verdict: Verdict[];
  evidence: (string | null)[];
}

export interface Replay {
  id: string;
  title?: string;
  description?: string;
  is_demo_data?: boolean;
  system: string;
  L: number;
  dt: number;
  onset_step: number;
  perturbation: { type: string; subtype: string; severity: string };
  expected_verdict: Verdict;
  frames: Frames;
}

const NUMERIC: (keyof Frames)[] = ["t", "true_theta"];
const NULLABLE: (keyof Frames)[] = ["measured_theta", "pred_theta", "ai_confidence", "referee_score"];
const UNIT: (keyof Frames)[] = ["ai_confidence", "referee_score"];

const isNum = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const isVerdict = (x: unknown): x is Verdict => VERDICTS.includes(x as Verdict);

export class ReplayError extends Error {
  constructor(public source: string, public problems: string[]) {
    super(`${source}: ${problems.join("; ")}`);
  }
}

export function validateIndex(raw: unknown, source = "index.json"): ReplayIndex {
  const p: string[] = [];
  const o = raw as ReplayIndex;
  if (!o || typeof o !== "object") throw new ReplayError(source, ["not an object"]);
  if (typeof o.is_demo_data !== "boolean") p.push("is_demo_data must be true/false");
  if (!Array.isArray(o.scenarios) || o.scenarios.length === 0) p.push("scenarios must be a non-empty list");
  const ids = new Set<string>();
  (o.scenarios ?? []).forEach((s, i) => {
    if (!s.id || !s.file || !s.title) p.push(`scenarios[${i}] needs id, title and file`);
    if (ids.has(s.id)) p.push(`duplicate scenario id ${s.id}`);
    ids.add(s.id);
    if (!isVerdict(s.expected_verdict)) p.push(`scenarios[${i}].expected_verdict invalid`);
    if (!s.selection_rule) p.push(`scenarios[${i}] needs a selection_rule (honesty rule)`);
  });
  if (p.length) throw new ReplayError(source, p);
  return o;
}

export function validateReplay(raw: unknown, source = "replay"): Replay {
  const p: string[] = [];
  const r = raw as Replay;
  if (!r || typeof r !== "object" || !r.frames) throw new ReplayError(source, ["missing frames"]);
  if (!isNum(r.L) || r.L <= 0) p.push("L must be a positive number");
  if (!isNum(r.dt) || r.dt <= 0) p.push("dt must be a positive number");
  if (!isVerdict(r.expected_verdict)) p.push("expected_verdict invalid");

  const f = r.frames;
  const n = Array.isArray(f.t) ? f.t.length : 0;
  if (n < 2) p.push("frames.t needs at least 2 entries");
  const all = [...NUMERIC, ...NULLABLE, "verdict", "evidence"] as (keyof Frames)[];
  for (const k of all) {
    if (!Array.isArray(f[k])) p.push(`frames.${k} missing`);
    else if (f[k].length !== n) p.push(`frames.${k} has ${f[k].length} entries, expected ${n}`);
  }
  if (p.length) throw new ReplayError(source, p);

  for (const k of NUMERIC) if (!(f[k] as unknown[]).every(isNum)) p.push(`frames.${k} must be all numbers`);
  for (const k of NULLABLE) if (!(f[k] as unknown[]).every((x) => x === null || isNum(x))) p.push(`frames.${k} must be numbers or null`);
  for (const k of UNIT) if (!(f[k] as (number | null)[]).every((x) => x === null || (x >= 0 && x <= 1))) p.push(`frames.${k} must be within 0–1`);
  if (!f.verdict.every(isVerdict)) p.push("frames.verdict has an unknown verdict");
  if (!f.t.every((x, i) => i === 0 || x > f.t[i - 1])) p.push("frames.t must increase");
  if (!Number.isInteger(r.onset_step) || r.onset_step < -1 || r.onset_step >= n) p.push("onset_step must be -1 or a frame index");
  if (p.length) throw new ReplayError(source, p);
  return r;
}

async function fetchJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new ReplayError(url, [`HTTP ${res.status}`]);
  return res.json();
}

export const REPLAY_BASE = `${import.meta.env.BASE_URL}replays/`;

export async function loadIndex(base = REPLAY_BASE): Promise<ReplayIndex> {
  return validateIndex(await fetchJson(`${base}index.json`));
}

export async function loadReplay(entry: ScenarioEntry, base = REPLAY_BASE): Promise<Replay> {
  return validateReplay(await fetchJson(`${base}${entry.file}`), entry.file);
}

/** First frame where the referee's verdict is not "trust", or -1. */
export function flagFrame(r: Replay): number {
  return r.frames.verdict.findIndex((v) => v !== "trust");
}

/** Value at fractional frame position `pos`, linearly interpolated; null if either side is null. */
export function sample(arr: (number | null)[], pos: number): number | null {
  const i = Math.max(0, Math.min(arr.length - 1, Math.floor(pos)));
  const j = Math.min(arr.length - 1, i + 1);
  const a = arr[i], b = arr[j];
  if (a === null || b === null) return pos - i < 0.5 ? a : b;
  return a + (b - a) * (pos - i);
}

export const PLAIN_VERDICT: Record<Verdict, string> = {
  trust: "All clear",
  model_error: "AI is wrong",
  instrument_fault: "Sensor is wrong",
  discovery: "Something new",
};
