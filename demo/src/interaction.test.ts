// State and lifecycle tests (PR #70 review): out-of-order loads, recoverable
// errors, input before the first replay loads, and a flag that later clears.

import { afterEach, describe, expect, it, vi } from "vitest";
import { App, type Views } from "./app";
import { type Replay, type ReplayIndex, ReplayError, type ScenarioEntry, type Verdict } from "./replay";
import { Timeline } from "./timeline";
import { cardState } from "./verdict";

const entry = (id: string): ScenarioEntry => ({
  id, title: id.toUpperCase(), file: `${id}.json`, expected_verdict: "trust", selection_rule: "test",
});
const INDEX: ReplayIndex = { generated_from: "test", is_demo_data: true, scenarios: [entry("a"), entry("b")] };

function replay(id: string, verdict: Verdict[] = ["trust", "trust", "trust", "trust"], expected: Verdict = "trust"): Replay {
  const n = verdict.length;
  return {
    id, system: "pendulum", L: 1, dt: 0.05, onset_step: -1,
    perturbation: { type: "none", subtype: "none", severity: "none" }, expected_verdict: expected,
    frames: {
      t: verdict.map((_, i) => i * 0.05), true_theta: Array(n).fill(0), measured_theta: Array(n).fill(0),
      pred_theta: Array(n).fill(0), ai_confidence: Array(n).fill(0.9), referee_score: Array(n).fill(0),
      verdict, evidence: verdict.map((v) => (v === "trust" ? null : "why")),
    },
  };
}

function deferred<T>() {
  let resolve!: (v: T) => void, reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => ((resolve = res), (reject = rej)));
  return { promise, resolve, reject };
}

function recordingViews() {
  const log: string[] = [];
  const views: Views = {
    highlight: (id) => log.push(`highlight ${id}`),
    show: (e) => log.push(`show ${e.id}`),
    showError: (p) => log.push(`error ${p.join("|")}`),
    clearError: () => log.push("clear"),
    pause: () => log.push("pause"),
  };
  return { log, views };
}

describe("App: scenario selection", () => {
  it("the latest selection wins when loads resolve out of order", async () => {
    const pending = { a: deferred<Replay>(), b: deferred<Replay>() };
    const { log, views } = recordingViews();
    const app = new App(INDEX, (e) => pending[e.id as "a" | "b"].promise, views);

    const openA = app.open("a");
    const openB = app.open("b");
    pending.b.resolve(replay("b"));
    expect(await openB).toBe(true);
    pending.a.resolve(replay("a")); // slower, older request lands last
    expect(await openA).toBe(false);

    expect(app.current?.id).toBe("b");
    expect(log.filter((l) => l.startsWith("show"))).toEqual(["show b"]);
    expect(log.filter((l) => l.startsWith("highlight")).at(-1)).toBe("highlight b");
  });

  it("a failed load keeps the player usable and the next selection recovers", async () => {
    const { log, views } = recordingViews();
    const load = vi.fn(async (e: ScenarioEntry) => {
      if (e.id === "a") throw new ReplayError("a.json", ["HTTP 503"]);
      return replay(e.id);
    });
    const app = new App(INDEX, load, views);

    expect(await app.open("a")).toBe(false);
    expect(log).toEqual(["highlight a", "pause", "error a.json: HTTP 503"]);
    expect(await app.open("b")).toBe(true);
    expect(log.slice(3)).toEqual(["highlight b", "clear", "show b"]);
    // Retrying the failed one later works too (failures are not cached).
    load.mockImplementation(async (e) => replay(e.id));
    expect(await app.open("a")).toBe(true);
  });

  it("an older request failing late does not show an error over the current scenario", async () => {
    const pending = { a: deferred<Replay>(), b: deferred<Replay>() };
    const { log, views } = recordingViews();
    const app = new App(INDEX, (e) => pending[e.id as "a" | "b"].promise, views);
    const openA = app.open("a");
    const openB = app.open("b");
    pending.b.resolve(replay("b"));
    await openB;
    pending.a.reject(new Error("timeout"));
    expect(await openA).toBe(false);
    expect(log.some((l) => l.startsWith("error"))).toBe(false);
  });

  it("unknown ids fall back to the first scenario", async () => {
    const { views } = recordingViews();
    const app = new App(INDEX, async (e) => replay(e.id), views);
    await app.open("nope");
    expect(app.current?.id).toBe("a");
  });
});

describe("Timeline: input before the first replay loads", () => {
  afterEach(() => vi.unstubAllGlobals());

  function fakeTimeline() {
    const handlers: Record<string, (e: unknown) => void> = {};
    const el = (name: string) => ({
      disabled: true, value: "0", max: "0", textContent: "", style: {} as Record<string, string>,
      setAttribute: vi.fn(),
      addEventListener: (type: string, fn: (e: unknown) => void) => (handlers[`${name}:${type}`] = fn),
    });
    const scrub = el("scrub"), play = el("play"), icon = el("icon"), clock = el("clock");
    const marks = { onset: el("m1"), flag: el("m2"), onsetLabel: el("m3"), flagLabel: el("m4") };
    const keys = el("keys");
    const tl = new Timeline(scrub as never, play as never, icon as never, clock as never, marks as never, keys);
    return { tl, handlers, scrub, play, clock };
  }

  it("ignores play, Space and scrubbing until load(), then enables the controls", () => {
    vi.stubGlobal("requestAnimationFrame", vi.fn(() => 1));
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    const { tl, handlers, scrub, play, clock } = fakeTimeline();
    const space = { code: "Space", target: { tagName: "BODY" }, preventDefault: vi.fn(), shiftKey: false };

    expect(() => handlers["play:click"]({})).not.toThrow();
    expect(() => handlers["keys:keydown"](space)).not.toThrow();
    scrub.value = "5";
    expect(() => handlers["scrub:input"]({})).not.toThrow();
    expect(() => tl.play()).not.toThrow();
    expect(tl.playing).toBe(false);
    expect(space.preventDefault).not.toHaveBeenCalled();
    expect(scrub.disabled && play.disabled).toBe(true);

    const onFrame = vi.fn();
    tl.load({ frames: 10, dt: 0.05, onsetFrame: -1, flagFrame: -1, onFrame });
    expect(scrub.disabled || play.disabled).toBe(false);
    handlers["keys:keydown"](space);
    expect(tl.playing).toBe(true);
    scrub.value = "4";
    handlers["scrub:input"]({});
    expect(tl.playing).toBe(false);
    expect(tl.pos).toBe(4);
    expect(clock.textContent).toBe("0.20 s");
    expect(onFrame).toHaveBeenLastCalledWith(4);
  });
});

describe("verdict card: a flag that clears before the end", () => {
  it("reports the earlier flag at the last frame instead of 'Watching'", () => {
    const r = replay("x", ["trust", "instrument_fault", "trust", "trust"], "instrument_fault");
    expect(cardState(r, 1).kind).toBe("flagged");
    expect(cardState(r, 2).kind).toBe("watching");
    expect(cardState(r, 3)).toEqual({ kind: "end-flagged-earlier", verdict: "instrument_fault", evidence: "why", correct: true });
  });

  it("the outcome is the first flag, graded against the answer key", () => {
    const r = replay("x", ["trust", "model_error", "discovery", "trust"], "discovery");
    expect(cardState(r, 3)).toMatchObject({ kind: "end-flagged-earlier", verdict: "model_error", correct: false });
  });
});
