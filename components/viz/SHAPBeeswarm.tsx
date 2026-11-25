"use client";

import React, { useEffect, useRef } from "react";
import { Chip } from "@/ui/Chip";

export const SHAPBeeswarm: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = (canvas.width = canvas.parentElement?.clientWidth || 600);
    const height = (canvas.height = 240);

    const features = ["PM2.5", "CO", "PM10", "City"];
    const pointsPerFeature = 50;

    ctx.clearRect(0, 0, width, height);

    features.forEach((feat, rowIdx) => {
      const y = 40 + rowIdx * 50;

      // Axis Line
      ctx.beginPath();
      ctx.moveTo(100, y);
      ctx.lineTo(width - 40, y);
      ctx.strokeStyle = "#1f293d";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Feature Label
      ctx.fillStyle = "#f3f4f6";
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      ctx.fillText(feat, 20, y + 4);

      // Beeswarm Points
      for (let i = 0; i < pointsPerFeature; i++) {
        const val = (Math.random() - 0.5) * (width - 160);
        const px = width / 2 + val * (1 - rowIdx * 0.2);
        const py = y + (Math.random() - 0.5) * 18;

        const isPositive = val > 0;
        ctx.beginPath();
        ctx.arc(px, py, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = isPositive ? "rgba(239, 68, 68, 0.7)" : "rgba(59, 130, 246, 0.7)";
        ctx.fill();
      }
    });
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto bg-surface/70 border border-surface-border p-6 sm:p-8 rounded-3xl backdrop-blur-md space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <span className="text-xs font-mono text-amber-400 uppercase tracking-widest">
            SHAP Summary Beeswarm
          </span>
          <h4 className="text-xl font-bold text-white">Impact Distribution</h4>
        </div>
        <Chip status="illustrative" text="Illustrative Beeswarm" />
      </div>

      <div className="relative w-full overflow-hidden rounded-2xl bg-surface/40 border border-surface-border p-4">
        <canvas ref={canvasRef} className="w-full h-[240px]" />
      </div>
      <div className="flex justify-between text-xs font-mono text-gray-400">
        <span className="text-blue-400">← Decreases Predicted AQI</span>
        <span className="text-red-400">Increases Predicted AQI →</span>
      </div>
    </div>
  );
};
