import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { ReplayError, type Replay, flagFrame, sample, validateIndex, validateReplay } from "./replay";
import { rateAt } from "./timeline";
import { cardState } from "./verdict";

const DIR = join(__dirname, "../public/replays");
const read = (f: string) => JSON.parse(readFileSync(join(DIR, f), "utf8"));

function tiny(overrides: Partial<Replay["frames"]> = {}, top: Partial<Replay> = {}): Replay {
  return {
    id: "x", system: "pendulum", L: 1, dt: 0.05, onset_step: 2,
    perturbation: { type: "sensor_fault", subtype: "bias", severity: "med" },
    expected_verdict: "instrument_fault",
    frames: {
      t: [0, 0.05, 0.1, 0.15],
      true_theta: [0.1, 0.2, 0.3, 0.4],
      measured_theta: [0.1, 0.2, null, 0.5],
      pred_theta: [0.1, 0.2, 0.3, 0.3],
      ai_confidence: [0.9, 0.9, null, 0.9],
      referee_score: [0, 0.1, 0.4, 0.8],
      verdict: ["trust", "trust", "trust", "instrument_fault"],
      evidence: [null, null, null, "readings contradict"],
      ...overrides,
    },
    ...top,
  };
}

describe("shipped replays", () => {
  const index = validateIndex(read("index.json"));

  it("index lists every replay file, each with a selection rule", () => {
    const files = readdirSync(DIR).filter((f) => f !== "index.json").sort();
    expect(index.scenarios.map((s) => s.file).sort()).toEqual(files);
  });

  it.each(index.scenarios.map((s) => [s.id, s] as const))("%s validates and matches its index entry", (_, s) => {
    const r = validateReplay(read(s.file), s.file);
    expect(r.id).toBe(s.id);
    expect(r.expected_verdict).toBe(s.expected_verdict);
    expect(r.is_demo_data).toBe(index.is_demo_data);
  });

  it("covers all three causes, a clean run and an honest miss (plan §9)", () => {
    const verdicts = new Set(index.scenarios.map((s) => s.expected_verdict));
    expect(verdicts).toEqual(new Set(["trust", "model_error", "instrument_fault", "discovery"]));
    const missed = index.scenarios.filter((s) => s.missed);
    expect(missed.length).toBeGreaterThan(0);
    for (const s of missed) expect(flagFrame(validateReplay(read(s.file)))).toBe(-1);
  });
});

describe("validateReplay rejects bad files", () => {
  const bad: [string, Replay][] = [
    ["length mismatch", tiny({ pred_theta: [0.1, 0.2, 0.3] })],
    ["unknown verdict", tiny({ verdict: ["trust", "trust", "trust", "oops" as never] })],
    ["score above 1", tiny({ referee_score: [0, 0, 0, 1.5] })],
    ["null true_theta", tiny({ true_theta: [0.1, null as never, 0.3, 0.4] })],
    ["time not increasing", tiny({ t: [0, 0.1, 0.1, 0.2] })],
    ["onset out of range", tiny({}, { onset_step: 9 })],
    ["bad L", tiny({}, { L: 0 })],
  ];
  it.each(bad)("%s", (_, r) => {
    expect(() => validateReplay(r)).toThrow(ReplayError);
  });

  it("accepts a good one", () => {
    expect(validateReplay(tiny()).id).toBe("x");
  });

  it("index without selection rule fails", () => {
    const idx = { generated_from: "x", is_demo_data: true, scenarios: [{ id: "a", title: "A", file: "a.json", expected_verdict: "trust" }] };
    expect(() => validateIndex(idx)).toThrow(/selection_rule/);
  });
});

describe("helpers", () => {
  it("sample interpolates and respects nulls", () => {
    expect(sample([0, 1, 2], 0.5)).toBeCloseTo(0.5);
    expect(sample([0, null, 2], 0.2)).toBe(0);
    expect(sample([0, null, 2], 0.8)).toBe(null);
    expect(sample([0, 1], 5)).toBe(1);
  });

  it("slow-mo only near the flag, only when on", () => {
    expect(rateAt(100, 100, 0.05, true)).toBeLessThan(1);
    expect(rateAt(100 + 30, 100, 0.05, true)).toBe(1);
    expect(rateAt(100, 100, 0.05, false)).toBe(1);
    expect(rateAt(100, -1, 0.05, true)).toBe(1);
  });
});

describe("verdict card", () => {
  it("watching, then flagged with the answer key", () => {
    const r = tiny();
    expect(cardState(r, 1).kind).toBe("watching");
    expect(cardState(r, 3)).toEqual({ kind: "flagged", verdict: "instrument_fault", evidence: "readings contradict", correct: true });
  });

  it("wrong flag is marked incorrect", () => {
    const r = tiny({}, { expected_verdict: "discovery" });
    expect(cardState(r, 3)).toMatchObject({ kind: "flagged", correct: false });
  });

  it("a miss is only revealed at the end", () => {
    const r = tiny({ verdict: ["trust", "trust", "trust", "trust"], evidence: [null, null, null, null] });
    expect(cardState(r, 2).kind).toBe("watching");
    expect(cardState(r, 3)).toEqual({ kind: "end-clear", correct: false });
    const clean = tiny({ verdict: ["trust", "trust", "trust", "trust"] }, { expected_verdict: "trust" });
    expect(cardState(clean, 3)).toEqual({ kind: "end-clear", correct: true });
  });
});
