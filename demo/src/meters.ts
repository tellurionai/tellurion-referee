// Two meters side by side: "AI confidence" (from the model) vs "Referee" (disagreement score).

import { type Replay, sample } from "./replay";

class Meter {
  private fill: HTMLElement;
  private value: HTMLElement;

  constructor(private el: HTMLElement) {
    this.fill = el.querySelector(".meter-fill")!;
    this.value = el.querySelector(".meter-value")!;
  }

  set(v: number | null, flagged = false) {
    this.el.classList.toggle("flagged", flagged);
    if (v === null) {
      this.fill.style.width = "0";
      this.value.textContent = "–";
      return;
    }
    this.fill.style.width = `${(v * 100).toFixed(1)}%`;
    this.value.textContent = `${Math.round(v * 100)}%`;
  }
}

export class Meters {
  private ai: Meter;
  private ref: Meter;
  private replay: Replay | null = null;

  constructor(aiEl: HTMLElement, refEl: HTMLElement) {
    this.ai = new Meter(aiEl);
    this.ref = new Meter(refEl);
  }

  load(replay: Replay) {
    this.replay = replay;
  }

  show(pos: number) {
    const f = this.replay!.frames;
    const flagged = f.verdict[Math.floor(pos)] !== "trust";
    this.ai.set(sample(f.ai_confidence, pos));
    this.ref.set(sample(f.referee_score, pos), flagged);
  }
}
