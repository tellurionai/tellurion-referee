// Play / pause / scrub, onset + flag markers, real-time playback with optional
// slow-mo around the referee flag.

export interface TimelineOptions {
  frames: number;
  dt: number;
  onsetFrame: number; // -1 if none
  flagFrame: number; // -1 if none
  onFrame: (pos: number) => void;
}

const SLOWMO_RATE = 0.3;
const SLOWMO_WINDOW_S = 1.2; // each side of the flag

/** Playback rate at fractional frame `pos`: 1 = real time. */
export function rateAt(pos: number, flagFrame: number, dt: number, slowmo: boolean): number {
  if (!slowmo || flagFrame < 0) return 1;
  return Math.abs(pos - flagFrame) * dt <= SLOWMO_WINDOW_S ? SLOWMO_RATE : 1;
}

export class Timeline {
  pos = 0;
  playing = false;
  slowmo = true;
  private last = 0;
  private raf = 0;
  private opts!: TimelineOptions;

  constructor(
    private scrub: HTMLInputElement,
    private playBtn: HTMLButtonElement,
    private playIcon: SVGPathElement,
    private clock: HTMLElement,
    private marks: { onset: HTMLElement; flag: HTMLElement; onsetLabel: HTMLElement; flagLabel: HTMLElement },
  ) {
    playBtn.addEventListener("click", () => this.toggle());
    scrub.addEventListener("input", () => {
      this.pause();
      this.seek(Number(scrub.value));
    });
    document.addEventListener("keydown", (e) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" && (e.target as HTMLInputElement).type !== "range") return;
      if (e.code === "Space") {
        e.preventDefault();
        this.toggle();
      } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
        if (tag === "INPUT") return; // the range input handles its own arrows
        e.preventDefault();
        this.pause();
        this.seek(this.pos + (e.code === "ArrowRight" ? 1 : -1) * (e.shiftKey ? 10 : 1));
      }
    });
  }

  load(opts: TimelineOptions) {
    this.opts = opts;
    this.scrub.max = String(opts.frames - 1);
    this.place(this.marks.onset, this.marks.onsetLabel, opts.onsetFrame);
    this.place(this.marks.flag, this.marks.flagLabel, opts.flagFrame);
    this.pause();
    this.seek(0);
  }

  private place(mark: HTMLElement, label: HTMLElement, frame: number) {
    const show = frame >= 0;
    mark.style.display = label.style.display = show ? "block" : "none";
    if (!show) return;
    // Range thumbs inset the track by half a thumb (~8px) on each side.
    const pct = frame / (this.opts.frames - 1);
    const left = `calc(8px + (100% - 16px) * ${pct})`;
    mark.style.left = label.style.left = left;
  }

  seek(pos: number) {
    const max = this.opts.frames - 1;
    this.pos = Math.max(0, Math.min(max, pos));
    this.scrub.value = String(this.pos);
    this.clock.textContent = `${(this.pos * this.opts.dt).toFixed(2)} s`;
    this.opts.onFrame(this.pos);
  }

  toggle() {
    if (this.playing) this.pause();
    else this.play();
  }

  play() {
    if (this.pos >= this.opts.frames - 1) this.seek(0);
    this.playing = true;
    this.playBtn.setAttribute("aria-label", "Pause");
    this.playIcon.setAttribute("d", "M7 5h4v14H7zM13 5h4v14h-4z");
    this.last = performance.now();
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(this.tick);
  }

  pause() {
    this.playing = false;
    this.playBtn.setAttribute("aria-label", "Play");
    this.playIcon.setAttribute("d", "M8 5v14l11-7z");
    cancelAnimationFrame(this.raf);
  }

  private tick = (now: number) => {
    if (!this.playing) return;
    const elapsed = Math.min(0.1, (now - this.last) / 1000); // clamp after tab switches
    this.last = now;
    const { dt, flagFrame, frames } = this.opts;
    const next = this.pos + (elapsed / dt) * rateAt(this.pos, flagFrame, dt, this.slowmo);
    this.seek(next);
    if (next >= frames - 1) {
      this.pause();
      return;
    }
    this.raf = requestAnimationFrame(this.tick);
  };
}
