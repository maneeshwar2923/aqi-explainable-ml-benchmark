"use client";

import React from "react";

interface ChipProps {
  status: "reported" | "pending" | "illustrative";
  text?: string;
}

export const Chip: React.FC<ChipProps> = ({ status, text }) => {
  if (status === "reported") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
        {text || "Reported Data"}
      </span>
    );
  }

  if (status === "pending") {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30">
        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping" />
        {text || "Pending Source Data"}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono bg-purple-500/10 text-purple-400 border border-purple-500/30">
      <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
      {text || "Illustrative Concept"}
    </span>
  );
};
