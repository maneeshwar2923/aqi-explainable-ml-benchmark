"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { MODELS_DATA } from "@/data/models";
import { Trophy, HelpCircle } from "lucide-react";
import { useStoryStore } from "@/store/useStoryStore";

export const MetricBarRace: React.FC = () => {
  const [metricTab, setMetricTab] = useState<"rmse" | "mae" | "r2">("rmse");
  const { openExplanation } = useStoryStore();

  const getMetricValue = (m: (typeof MODELS_DATA)[0]) => {
    if (metricTab === "rmse") return m.rmse;
    if (metricTab === "mae") return m.mae;
    return m.r2;
  };

  const getMetricMax = () => {
    if (metricTab === "rmse") return 35;
    if (metricTab === "mae") return 25;
    return 1.0;
  };

  return (
    <div className="w-full max-w-4xl mx-auto bg-surface/70 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md space-y-8">
      {/* Metric Selector Tabs */}
      <div className="flex items-center justify-between border-b border-surface-border pb-4">
        <div className="flex gap-2">
          {(["rmse", "mae", "r2"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setMetricTab(tab)}
              className={`px-4 py-2 rounded-xl font-mono text-xs uppercase tracking-wider transition-all ${
                metricTab === tab
                  ? "bg-amber-500 text-black font-bold shadow-lg shadow-amber-500/20"
                  : "bg-surface/50 text-gray-400 hover:text-white border border-surface-border"
              }`}
            >
              {tab.toUpperCase()} Benchmark
            </button>
          ))}
        </div>

        <button
          onClick={() => openExplanation(metricTab.toUpperCase())}
          className="flex items-center gap-1.5 text-xs font-mono text-amber-400 hover:underline"
        >
          <HelpCircle className="w-3.5 h-3.5" /> What is {metricTab.toUpperCase()}?
        </button>
      </div>

      {/* Horizontal Bar Race */}
      <div className="space-y-6">
        {MODELS_DATA.map((model) => {
          const val = getMetricValue(model);
          const max = getMetricMax();
          const percent = val ? (val / max) * 100 : 0;

          return (
            <div key={model.id} className="space-y-2">
              <div className="flex items-center justify-between text-sm font-semibold">
                <div className="flex items-center gap-2 text-white">
                  {model.name}
                  {model.isBest && (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-mono">
                      <Trophy className="w-3 h-3" /> Winner
                    </span>
                  )}
                </div>
                <div className="font-mono text-sm">
                  {val !== null ? (
                    <span className={model.isBest ? "text-amber-400 font-bold" : "text-gray-300"}>
                      {val}
                    </span>
                  ) : (
                    <span className="text-xs text-amber-400/80 font-mono">
                      Not reported in source data
                    </span>
                  )}
                </div>
              </div>

              <div className="h-4 w-full bg-surface-border/50 rounded-full overflow-hidden relative">
                {val !== null ? (
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${percent}%` }}
                    viewport={{ once: true }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                    className={`h-full rounded-full transition-all ${
                      model.isBest
                        ? "bg-gradient-to-r from-amber-500 to-yellow-300 glow-gold"
                        : "bg-gradient-to-r from-blue-500 to-indigo-400"
                    }`}
                  />
                ) : (
                  <div className="h-full w-full bg-amber-500/10 animate-shimmer rounded-full border border-amber-500/20 flex items-center justify-center">
                    <span className="text-[10px] font-mono text-amber-400/70 tracking-widest uppercase">
                      Pending Source Benchmark
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
