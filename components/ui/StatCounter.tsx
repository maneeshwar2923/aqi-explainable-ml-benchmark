"use client";

import React, { useEffect, useState } from "react";
import { useInView } from "framer-motion";

interface StatCounterProps {
  value: number;
  label: string;
  suffix?: string;
  prefix?: string;
  duration?: number;
}

export const StatCounter: React.FC<StatCounterProps> = ({
  value,
  label,
  suffix = "",
  prefix = "",
  duration = 1.5,
}) => {
  const ref = React.useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: "-50px" });
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!isInView) return;

    let start = 0;
    const end = value;
    const totalSteps = 40;
    const stepTime = (duration * 1000) / totalSteps;
    const increment = (end - start) / totalSteps;

    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [isInView, value, duration]);

  return (
    <div ref={ref} className="bg-surface/60 border border-surface-border p-6 rounded-2xl flex flex-col justify-between hover:border-amber-500/30 transition-all duration-300">
      <div className="font-mono text-4xl sm:text-5xl font-extrabold text-white tracking-tight mb-2">
        {prefix}
        {count.toLocaleString()}
        {suffix}
      </div>
      <div className="text-xs sm:text-sm font-sans uppercase tracking-wider text-gray-400 font-medium">
        {label}
      </div>
    </div>
  );
};
