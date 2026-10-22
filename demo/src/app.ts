// Scenario selection without the DOM: which replay is shown, in what order loads
// may land, and how a failed load is reported. main.ts supplies the views.

import { type Replay, ReplayError, type ReplayIndex, type ScenarioEntry } from "./replay";

export interface Views {
  /** Mark `id` as the selected chip. Called as soon as the user picks it. */
  highlight(id: string): void;
  /** Show a loaded replay: title, stage, meters, card, timeline. */
  show(entry: ScenarioEntry, replay: Replay): void;
  /** Report a load failure without removing the player. */
  showError(problems: string[]): void;
  clearError(): void;
  /** Stop playback (used when a load fails). */
  pause(): void;
}

/** Scenario id from a location hash like "#scenario=hidden-magnet"; the first scenario otherwise. */
export function idFromHash(hash: string, index: ReplayIndex): string {
  const id = hash.match(/scenario=([\w-]+)/)?.[1];
  return index.scenarios.some((s) => s.id === id) ? id! : index.scenarios[0].id;
}

export function problemsOf(err: unknown): string[] {
  if (err instanceof ReplayError) return err.problems.map((p) => `${err.source}: ${p}`);
  return [err instanceof Error ? err.message : String(err)];
}

export class App {
  current: Replay | null = null;
  entry: ScenarioEntry | null = null;
  private latest = 0;
  private cache = new Map<string, Replay>();

  constructor(
    private index: ReplayIndex,
    private load: (entry: ScenarioEntry) => Promise<Replay>,
    private views: Views,
  ) {}

  /**
   * Select a scenario. Only the most recent call may change what is shown: a
   * slower earlier load that resolves later is discarded. Returns true if this
   * call's replay is now on screen.
   */
  async open(id: string): Promise<boolean> {
    const entry = this.index.scenarios.find((s) => s.id === id) ?? this.index.scenarios[0];
    const request = ++this.latest;
    this.views.highlight(entry.id);
    let replay = this.cache.get(entry.id);
    if (!replay) {
      try {
        replay = await this.load(entry);
      } catch (err) {
        if (request === this.latest) {
          this.views.pause();
          this.views.showError(problemsOf(err));
        }
        return false;
      }
      this.cache.set(entry.id, replay);
    }
    if (request !== this.latest) return false;
    this.entry = entry;
    this.current = replay;
    this.views.clearError();
    this.views.show(entry, replay);
    return true;
  }
}
