// Pendulum canvas: reality (solid), AI prediction (brass ghost + fading trail),
// sensor reading (small ring; only moves off reality when the sensor is wrong).
// Positions are computed here: x = L sin θ, y = −L cos θ, pivot at the top.

import { type Replay, sample } from "./replay";

const TRAIL_S = 1.5;

interface Palette {
  reality: string;
  ai: string;
  sensor: string;
  muted: string;
  line: string;
  ink: string;
}

function palette(): Palette {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim();
  return { reality: v("--reality"), ai: v("--ai"), sensor: v("--sensor"), muted: v("--muted"), line: v("--line"), ink: v("--ink") };
}

export class Stage {
  private ctx: CanvasRenderingContext2D;
  private replay: Replay | null = null;
  private pos = 0;
  private colors = palette();

  constructor(private canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    new ResizeObserver(() => this.draw()).observe(canvas);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.colors = palette();
      this.draw();
    });
  }

  load(replay: Replay) {
    this.replay = replay;
    this.pos = 0;
    this.draw();
  }

  show(pos: number) {
    this.pos = pos;
    this.draw();
  }

  private draw() {
    const { canvas, ctx, replay } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!replay) return;

    const c = this.colors;
    const f = replay.frames;
    // Room for full rotations: pivot in the middle, string ~40% of the short side.
    const px = w / 2, py = h * 0.46;
    const R = Math.min(w, h) * 0.4;
    const at = (theta: number) => ({ x: px + R * Math.sin(theta), y: py + R * Math.cos(theta) });
    const bob = Math.max(9, R * 0.075);

    if (replay.is_demo_data) this.watermark(w, h);

    // Swing circle + pivot mount.
    ctx.setLineDash([3, 6]);
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(px, py, R, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = c.muted;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(px - 26, py);
    ctx.lineTo(px + 26, py);
    ctx.stroke();

    // AI ghost trail: the last TRAIL_S seconds of predicted bob positions, fading out.
    const steps = Math.round(TRAIL_S / replay.dt);
    const end = Math.floor(this.pos);
    ctx.lineCap = "round";
    ctx.lineWidth = 3;
    for (let k = Math.max(1, end - steps); k <= end; k++) {
      const a = f.pred_theta[k - 1], b = f.pred_theta[k];
      if (a === null || b === null) continue;
      const p0 = at(a), p1 = at(b);
      ctx.globalAlpha = 0.45 * (1 - (end - k) / steps);
      ctx.strokeStyle = c.ai;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // AI ghost.
    const pred = sample(f.pred_theta, this.pos);
    if (pred !== null) this.pendulum(px, py, at(pred), bob, c.ai, 0.6);

    // Reality.
    const truth = sample(f.true_theta, this.pos)!;
    this.pendulum(px, py, at(truth), bob, c.reality, 1);

    // Sensor reading.
    const meas = sample(f.measured_theta, this.pos);
    if (meas !== null) {
      const m = at(meas);
      ctx.strokeStyle = c.sensor;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(m.x, m.y, bob * 0.55, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.fillStyle = c.sensor;
      ctx.font = "600 13px Inter, system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("Sensor: no reading", 12, h - 12);
    }
  }

  private pendulum(px: number, py: number, p: { x: number; y: number }, r: number, color: string, alpha: number) {
    const { ctx } = this;
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  private watermark(w: number, h: number) {
    const { ctx } = this;
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.rotate(-Math.PI / 9);
    ctx.globalAlpha = 0.07;
    ctx.fillStyle = this.colors.ink;
    ctx.font = `700 ${Math.round(Math.min(w, h) * 0.16)}px Inter, system-ui, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("DEMO DATA", 0, 0);
    ctx.restore();
  }
}
