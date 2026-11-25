"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";

interface CityNode {
  name: string;
  x: number; // percentage in SVG viewBox
  y: number; // percentage in SVG viewBox
  severity: "high" | "moderate" | "low";
}

const CITIES: CityNode[] = [
  { name: "Ahmedabad", x: 26, y: 48, severity: "high" },
  { name: "Aizawl", x: 88, y: 45, severity: "low" },
  { name: "Amaravati", x: 44, y: 68, severity: "moderate" },
  { name: "Amritsar", x: 30, y: 22, severity: "high" },
  { name: "Bengaluru", x: 38, y: 76, severity: "low" },
  { name: "Bhopal", x: 40, y: 46, severity: "moderate" },
  { name: "Brajrajnagar", x: 62, y: 49, severity: "moderate" },
  { name: "Chandigarh", x: 34, y: 24, severity: "high" },
  { name: "Chennai", x: 46, y: 78, severity: "moderate" },
  { name: "Coimbatore", x: 36, y: 82, severity: "low" },
  { name: "Delhi", x: 36, y: 30, severity: "high" },
  { name: "Ernakulam", x: 34, y: 85, severity: "low" },
  { name: "Gurugram", x: 35, y: 31, severity: "high" },
  { name: "Guwahati", x: 82, y: 40, severity: "moderate" },
  { name: "Hyderabad", x: 42, y: 62, severity: "moderate" },
  { name: "Jaipur", x: 32, y: 36, severity: "high" },
  { name: "Jorapokhar", x: 68, y: 44, severity: "high" },
  { name: "Kochi", x: 34, y: 86, severity: "low" },
  { name: "Kolkata", x: 72, y: 48, severity: "high" },
  { name: "Lucknow", x: 48, y: 36, severity: "high" },
  { name: "Mumbai", x: 24, y: 60, severity: "moderate" },
  { name: "Patna", x: 62, y: 40, severity: "high" },
  { name: "Shillong", x: 81, y: 42, severity: "low" },
  { name: "Talcher", x: 64, y: 52, severity: "high" },
  { name: "Thiruvananthapuram", x: 35, y: 90, severity: "low" },
  { name: "Visakhapatnam", x: 52, y: 62, severity: "moderate" },
];

export const CityMap: React.FC = () => {
  const [activeCity, setActiveCity] = useState<CityNode | null>(null);

  const getCityColor = (sev: string) => {
    if (sev === "high") return "#ef4444";
    if (sev === "moderate") return "#f97316";
    return "#10b981";
  };

  return (
    <div className="relative w-full max-w-xl mx-auto bg-surface/70 border border-surface-border p-6 rounded-3xl backdrop-blur-md overflow-hidden">
      <div className="flex items-center justify-between mb-4">
        <div>
          <span className="text-xs font-mono text-amber-400 uppercase tracking-wider">
            Spatial Distribution
          </span>
          <h4 className="text-xl font-bold text-white">26 Monitoring Cities</h4>
        </div>
        <div className="flex gap-3 text-xs font-mono">
          <span className="flex items-center gap-1 text-red-400">
            <span className="w-2 h-2 rounded-full bg-red-500" /> High AQI
          </span>
          <span className="flex items-center gap-1 text-orange-400">
            <span className="w-2 h-2 rounded-full bg-orange-500" /> Med AQI
          </span>
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500" /> Low AQI
          </span>
        </div>
      </div>

      <div className="relative w-full aspect-[4/5] bg-surface/40 rounded-2xl border border-surface-border/50 p-4">
        {/* Stylized India Subcontinent Path SVG */}
        <svg viewBox="0 0 100 100" className="w-full h-full">
          <path
            d="M 30,12 L 45,15 L 55,20 L 70,25 L 85,35 L 90,45 L 80,50 L 75,55 L 70,60 L 55,75 L 45,85 L 35,95 L 30,85 L 25,65 L 18,52 L 20,40 L 25,25 Z"
            fill="none"
            stroke="#1f293d"
            strokeWidth="1.5"
            strokeDasharray="2 2"
          />

          {/* City Pulsing Nodes */}
          {CITIES.map((city, idx) => {
            const color = getCityColor(city.severity);
            const isHovered = activeCity?.name === city.name;
            return (
              <g key={idx} onClick={() => setActiveCity(city)} className="cursor-pointer">
                <circle
                  cx={city.x}
                  cy={city.y}
                  r={isHovered ? 4.5 : 2.5}
                  fill={color}
                  className="transition-all duration-300"
                />
                <circle
                  cx={city.x}
                  cy={city.y}
                  r={isHovered ? 8 : 5}
                  fill="none"
                  stroke={color}
                  strokeWidth="0.8"
                  opacity="0.6"
                  className="animate-ping"
                  style={{ animationDuration: `${2 + (idx % 3)}s` }}
                />
              </g>
            );
          })}
        </svg>

        {/* Hover / Click Callout */}
        {activeCity ? (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute bottom-4 left-4 right-4 bg-surface/90 backdrop-blur-md p-4 rounded-xl border border-amber-500/30 flex items-center justify-between"
          >
            <div>
              <div className="font-bold text-white text-base">{activeCity.name}</div>
              <div className="text-xs text-gray-400 font-mono">
                Part of 26-city dataset (2015-2020)
              </div>
            </div>
            <span
              className="px-2.5 py-1 rounded-full text-xs font-mono font-semibold"
              style={{
                backgroundColor: getCityColor(activeCity.severity) + "20",
                color: getCityColor(activeCity.severity),
              }}
            >
              {activeCity.severity.toUpperCase()}
            </span>
          </motion.div>
        ) : (
          <div className="absolute bottom-4 left-4 right-4 text-center text-xs font-mono text-gray-500">
            Click or tap any glowing city node to view details
          </div>
        )}
      </div>
    </div>
  );
};
