import Lenis from "lenis";
import { gsap, ScrollTrigger, initGSAP } from "./gsapConfig";

let lenisInstance: Lenis | null = null;

export const initLenis = () => {
  if (typeof window === "undefined") return null;

  initGSAP();

  if (lenisInstance) return lenisInstance;

  lenisInstance = new Lenis({
    duration: 1.2,
    easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
    smoothWheel: true,
  });

  lenisInstance.on("scroll", ScrollTrigger.update);

  gsap.ticker.add((time) => {
    lenisInstance?.raf(time * 1000);
  });

  gsap.ticker.lagSmoothing(0);

  return lenisInstance;
};

export const getLenis = () => lenisInstance;
