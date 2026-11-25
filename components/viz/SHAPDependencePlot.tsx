"use client";

import React, { useState } from "react";
import { Chip } from "@/ui/Chip";

export const SHAPDependencePlot: React.FC = () => {
  const [pm25Val, setPm25Val] = useState<number>(120);

  // Compute reactive SHAP value estimate
  const shapEffect = (pm25Val - 50) * 0.45;

  return (
    <div className="w-full max-w-4xl mx-auto bg-surface/70 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md space-y-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-surface-border pb-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-amber-400 uppercase tracking-widest">
              Interactive Feature Dependence
            </span>
            <Chip status="illustrative" text="Interactive Simulation" />
          </div>
          <h4 className="text-xl font-bold text-white mt-1">
            PM2.5 Concentration vs. SHAP Impact
          </h4>
        </div>
        <div className="font-mono text-sm text-right">
          <div className="text-xs text-gray-400">Predicted Marginal SHAP</div>
          <div className="text-amber-400 font-bold">
            {shapEffect >= 0 ? `+${shapEffect.toFixed(1)}` : shapEffect.toFixed(1)} AQI pts
          </div>
        </div>
      </div>

      {/* Interactive Slider */}
      <div className="space-y-2 bg-surface/40 p-4 rounded-2xl border border-surface-border">
        <div className="flex justify-between text-xs font-mono text-gray-300">
          <span>PM2.5 Raw Level: <strong className="text-amber-400">{pm25Val} µg/m³</strong></span>
          <span>Max Range: 300 µg/m³</span>
        </div>
        <input
          type="range"
          min="10"
          max="300"
          value={pm25Val}
          onChange={(e) => setPm25Val(Number(e.target.value))}
          className="w-full accent-amber-500 bg-surface-border h-2 rounded-lg cursor-pointer"
        />
      </div>

      {/* Reactive Scatter Trend Visualization */}
      <div className="relative h-48 w-full bg-surface/80 border border-surface-border rounded-2xl p-4 flex items-end">
        <svg viewBox="0 0 400 120" className="w-full h-full overflow-visible">
          {/* Axis */}
          <line x1="20" y1="100" x2="380" y2="100" stroke="#1f293d" strokeWidth="2" />
          <line x1="20" y1="10" x2="20" y2="100" stroke="#1f293d" strokeWidth="2" />

          {/* Non-linear SHAP Dependence Curve */}
          <path
            d="M 20 95 Q 120 70 200 40 T 380 15"
            fill="none"
            stroke="#f97316"
            strokeWidth="3"
          />

          {/* Active Slider Indicator Pin */}
          <circle
            cx={20 + (pm25Val / 300) * 360}
            cy={95 - (pm25Val / 300) * 80}
            r="7"
            fill="#eab308"
            className="animate-pulse"
          />
        </svg>
      </div>
    </div>
  );
};
