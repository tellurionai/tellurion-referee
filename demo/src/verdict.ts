// Verdict card: quiet until the referee flags, then the one big animation in the
// demo (slide-in + pulse) with the cause and one line of evidence. The answer key
// is shown after the flag, or at the end of a run the referee never flagged.

import { PLAIN_VERDICT, type Replay, flagFrame } from "./replay";

export type CardState =
  | { kind: "watching" }
  | { kind: "flagged"; verdict: string; evidence: string; correct: boolean }
  | { kind: "end-clear"; correct: boolean };

/** What the card should say at fractional frame `pos`. Pure, so it is unit-tested. */
export function cardState(r: Replay, pos: number): CardState {
  const k = Math.floor(pos);
  const v = r.frames.verdict[k];
  if (v !== "trust") {
    return { kind: "flagged", verdict: v, evidence: r.frames.evidence[k] ?? "", correct: v === r.expected_verdict };
  }
  if (k >= r.frames.t.length - 1 && flagFrame(r) < 0) {
    return { kind: "end-clear", correct: r.expected_verdict === "trust" };
  }
  return { kind: "watching" };
}

export class VerdictCard {
  private kicker: HTMLElement;
  private title: HTMLElement;
  private evidence: HTMLElement;
  private key: HTMLElement;
  private replay: Replay | null = null;
  private lastKind = "";

  constructor(private el: HTMLElement) {
    this.kicker = el.querySelector("#verdict-kicker")!;
    this.title = el.querySelector("#verdict-title")!;
    this.evidence = el.querySelector("#verdict-evidence")!;
    this.key = el.querySelector("#verdict-key")!;
    el.addEventListener("animationend", () => el.classList.remove("pulse"));
  }

  load(replay: Replay) {
    this.replay = replay;
    this.lastKind = "";
  }

  show(pos: number) {
    const r = this.replay!;
    const s = cardState(r, pos);
    const answer = PLAIN_VERDICT[r.expected_verdict];
    this.el.classList.toggle("flagged", s.kind === "flagged");
    this.el.classList.toggle("missed", s.kind === "end-clear" && !s.correct);

    if (s.kind === "flagged") {
      if (this.lastKind !== "flagged") {
        this.el.classList.remove("pulse");
        void this.el.offsetWidth; // restart the animation
        this.el.classList.add("pulse");
      }
      this.kicker.textContent = "Referee flag";
      this.title.textContent = PLAIN_VERDICT[s.verdict as keyof typeof PLAIN_VERDICT];
      this.evidence.textContent = s.evidence;
      this.key.textContent = s.correct ? `Answer key: ${answer} ✓` : `Answer key says: ${answer} ✗`;
    } else if (s.kind === "end-clear") {
      this.kicker.textContent = "Referee";
      this.title.textContent = "No flag";
      this.evidence.textContent = s.correct
        ? "The AI and the sensors agreed the whole way. Nothing to report."
        : "The referee never flagged this run.";
      this.key.textContent = s.correct
        ? `Answer key: ${answer} ✓`
        : `Answer key says: ${answer}. This is a miss, shown on purpose.`;
    } else {
      this.kicker.textContent = "Referee";
      this.title.textContent = "Watching";
      this.evidence.textContent = "No disagreement it can't explain, so far.";
      this.key.textContent = "";
    }
    this.lastKind = s.kind;
  }
}
