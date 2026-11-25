import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

let isRegistered = false;

export const initGSAP = () => {
  if (typeof window !== "undefined" && !isRegistered) {
    gsap.registerPlugin(ScrollTrigger);
    gsap.defaults({
      ease: "power2.out",
      duration: 0.8,
    });
    isRegistered = true;
  }
};

export { gsap, ScrollTrigger };
