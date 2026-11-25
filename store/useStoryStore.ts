import { create } from "zustand";

interface StoryState {
  currentChapter: number;
  scrollProgress: number;
  explainMode: boolean;
  activeExplanationKey: string | null;
  setCurrentChapter: (chapter: number) => void;
  setScrollProgress: (progress: number) => void;
  toggleExplainMode: () => void;
  openExplanation: (key: string) => void;
  closeExplanation: () => void;
}

export const useStoryStore = create<StoryState>((set) => ({
  currentChapter: 1,
  scrollProgress: 0,
  explainMode: false,
  activeExplanationKey: null,
  setCurrentChapter: (chapter) => set({ currentChapter: chapter }),
  setScrollProgress: (progress) => set({ scrollProgress: progress }),
  toggleExplainMode: () => set((state) => ({ explainMode: !state.explainMode })),
  openExplanation: (key) => set({ activeExplanationKey: key }),
  closeExplanation: () => set({ activeExplanationKey: null }),
}));
