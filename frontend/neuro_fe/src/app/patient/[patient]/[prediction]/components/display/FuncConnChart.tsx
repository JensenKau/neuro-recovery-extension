"use client";

import React, { useState, useEffect } from "react";
import convert from 'color-convert';

interface FuncConnChartProps {
  matrix: Array<Array<number>>;
}

const FuncConnChart = ({matrix}: FuncConnChartProps) => {
  const [Chart, setChart] = useState<any>();

  const chartProperty = (() => {
    const regions = [
      "Fp1", "Fp2", "F7", "F8", "F3", "F4", "T3", "T4", "C3", "C4", "T5",
      "T6", "P3", "P4", "O1", "O2", "Fz", "Cz", "Pz", "Fpz", "Oz", "F9"
    ];

    const series =
      matrix.map((row, i) => {
        return {
          name: regions[i],
          data: row.map((col, j) => {
            return { x: regions[j], y: col }
          })
        };
      });

    const colorRangeNegative =
      Array.from({ length: 100 }, (val, index) => index - 100).map((val) => {
        return {
          from: val / 100,
          to: ((val + 1) / 100) - 1e-9,
          color: `#${convert.hwb.hex([240, val + 100, 0])}`
        };
      });

    const colorRangePositive =
      Array.from({ length: 100 }, (val, index) => index).map((val) => {
        return {
          from: val / 100,
          to: ((val + 1) / 100) - 1e-9,
          color: `#${convert.hwb.hex([0, 100 - val, 0])}`
        };
      });

    const colorRangeCombined = [
      ...colorRangeNegative,
      ...colorRangePositive,
      {
        from: 1,
        to: 1,
        color: `#${convert.hwb.hex([0, 0, 0])}`
      }
    ];

    return {
      series: series,
      legend: { show: false },
      dataLabels: { enabled: false },
      stroke: { width: 1 },
      yaxis: { reversed: true },
      plotOptions: {
        heatmap: {
          colorScale: {
            ranges: colorRangeCombined
          }
        }
      }
    };
  })();

  useEffect(() => {
    import("react-apexcharts").then((mod) => {
      setChart(() => mod.default);
    });
  }, []);

  return (
    <>
      {Chart && <Chart options={chartProperty} series={chartProperty.series} type="heatmap" />}
    </>
  );
};

export default FuncConnChart;