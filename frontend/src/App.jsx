import React, { useState } from "react";
import { Upload, Play, RefreshCw, Eye, Grid3x3, Box, AlertCircle, CheckCircle, Loader, X, Activity } from "lucide-react";
import LungIcon from "./assets/lung.svg";


const API_URL = "http://localhost:8000";

export default function LungNoduleApp() {
  const [scanId, setScanId] = useState(null);
  const [mhdFile, setMhdFile] = useState(null);
  const [rawFile, setRawFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [error, setError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState({ mhd: false, raw: false });
  const [scoreThreshold, setScoreThreshold] = useState(0.3);
  const [overlapStrategy, setOverlapStrategy] = useState("highest_score");
  const [results, setResults] = useState(null);
  const [selectedView, setSelectedView] = useState("axial");
  const [selectedSlice, setSelectedSlice] = useState(0);
  const [selectedInstanceId, setSelectedInstanceId] = useState(null);
  const [showMontage, setShowMontage] = useState(false);
  const [show3D, setShow3D] = useState(false);

  // ── EVALUATION STATE (NEW) ──
  const [annotationsLoaded, setAnnotationsLoaded] = useState(false);
  const [annotationsInfo, setAnnotationsInfo] = useState(null);
  const [seriesuid, setSeriesuid] = useState("");
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalResults, setEvalResults] = useState(null);
  const [showEvalSection, setShowEvalSection] = useState(false);

  const validateFile = (file, ext) =>
    file && file.name.toLowerCase().endsWith(ext);

  const handleFileUpload = async (file, type) => {
    if (!file) return;

    const ext = type === "mhd" ? ".mhd" : ".raw";
    if (!validateFile(file, ext)) {
      setError(`Invalid file. Expected ${ext}`);
      return;
    }

    try {
      setIsUploading(true);
      setError(null);

      const formData = new FormData();
      formData.append("file", file);
      if (type === "raw" && scanId) formData.append("scan_id", scanId);

      const res = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();

      if (type === "mhd") {
        setScanId(data.scan_id);
        setMhdFile(file);
        setUploadStatus({ mhd: true, raw: false });
        setResults(null);
        setEvalResults(null);
        // Auto-extract seriesuid from mhd filename
        const stem = file.name.replace(/\.mhd$/i, "");
        setSeriesuid(stem);
      } else {
        setRawFile(file);
        setUploadStatus((p) => ({ ...p, raw: true }));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const runInference = async () => {
    if (!scanId || !uploadStatus.mhd || !uploadStatus.raw) {
      setError("Upload both .mhd and .raw files");
      return;
    }

    setIsInferring(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("scan_id", scanId);
      formData.append("score_threshold", scoreThreshold);
      formData.append("overlap_strategy", overlapStrategy);

      const res = await fetch(`${API_URL}/infer`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Inference failed");

      const data = await res.json();
      setResults(data);
      setSelectedSlice(Math.floor(data.image_shape[0] / 2));
      setEvalResults(null); // Clear old eval when re-running
    } catch (err) {
      setError(err.message);
    } finally {
      setIsInferring(false);
    }
  };

  const cleanup = async () => {
    if (scanId) await fetch(`${API_URL}/cleanup/${scanId}`, { method: "DELETE" });
    setScanId(null);
    setMhdFile(null);
    setRawFile(null);
    setUploadStatus({ mhd: false, raw: false });
    setResults(null);
    setSelectedInstanceId(null);
    setError(null);
    setShowMontage(false);
    setShow3D(false);
    setEvalResults(null);
    setSeriesuid("");
  };

  const getConfidenceColor = (confidence) => {
  if (confidence >= 0.7) return "from-green-500 to-emerald-600";
  if (confidence >= 0.5) return "from-yellow-400 to-amber-500";
  return "from-red-500 to-rose-600";
};



  const getRiskLevel = (confidence) => {
    if (confidence >= 0.7) return "High Confidence";
    if (confidence >= 0.5) return "Moderate";
    return "Low Confidence";
  };

  const getMaxSlice = () => {
  if (!results) return 0;

  if (selectedView === "axial") {
    return results.image_shape[0] - 1; // D
  }
  if (selectedView === "coronal") {
    return results.image_shape[1] - 1; // H
  }
  if (selectedView === "sagittal") {
    return results.image_shape[2] - 1; // W
  }
  return 0;
};

const getSliceFromCentroid = (centroid) => {
  if (!centroid) return 0;

  if (selectedView === "axial") return Math.round(centroid[0]);
  if (selectedView === "coronal") return Math.round(centroid[1]);
  if (selectedView === "sagittal") return Math.round(centroid[2]);

  return 0;
};

const build3DUrl = () => {
  if (!scanId) return "";

  const base = `${API_URL}/visualize/${scanId}/3d`;
  const params = new URLSearchParams();

  if (selectedInstanceId) {
    params.append("instance_id", selectedInstanceId);
  }

  params.append("t", Date.now()); // cache buster
  return `${base}?${params.toString()}`;
};

const getSafeSlice = () => {
  if (!results) return 0;
  return Math.min(selectedSlice, getMaxSlice());
};

  // ── EVALUATION FUNCTIONS (NEW) ──

  const handleAnnotationsUpload = async (file) => {
    if (!file) return;
    try {
      setError(null);
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_URL}/eval/load_annotations`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Failed to load annotations");

      const data = await res.json();
      setAnnotationsLoaded(true);
      setAnnotationsInfo(data);
    } catch (err) {
      setError(`Annotations error: ${err.message}`);
    }
  };

  const runEvaluation = async () => {
    if (!scanId || !annotationsLoaded || !seriesuid) {
      setError("Need: inference completed + annotations loaded + valid seriesuid");
      return;
    }

    setIsEvaluating(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("seriesuid", seriesuid);
      formData.append("distance_thresh_mm", "15.0");
      formData.append("iou_thresh_match", "0.1");

      const res = await fetch(`${API_URL}/eval/evaluate_scan/${scanId}`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Evaluation failed");
      }

      const data = await res.json();
      setEvalResults(data);
    } catch (err) {
      setError(`Evaluation error: ${err.message}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  // Format metric value for display
  const fmtMetric = (val, decimals = 3) => {
    if (val === null || val === undefined) return "—";
    return Number(val).toFixed(decimals);
  };

  const fmtPct = (val) => {
    if (val === null || val === undefined) return "—";
    return `${(Number(val) * 100).toFixed(1)}%`;
  };

  // Get color class for a 0-1 metric
  const metricColor = (val) => {
    if (val === null || val === undefined) return "text-gray-400";
    if (val >= 0.7) return "text-green-400";
    if (val >= 0.4) return "text-yellow-400";
    return "text-red-400";
  };

  // Get bg gradient for metric cards
  const metricCardBg = (val) => {
    if (val === null || val === undefined) return "from-gray-500/10 to-gray-600/10 border-gray-500/20";
    if (val >= 0.7) return "from-green-500/15 to-emerald-500/15 border-green-400/25";
    if (val >= 0.4) return "from-yellow-500/15 to-amber-500/15 border-yellow-400/25";
    return "from-red-500/15 to-rose-500/15 border-red-400/25";
  };


  return (
    <div className="min-h-screen bg-slate-950 text-white">

      {/* Header */}
      <div className="bg-slate-950/90 backdrop-blur-sm border-b border-slate-800">

        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl flex items-center justify-center">
  <img
  src={LungIcon}
  alt="Lung Logo"
  className="w-7 h-7 object-contain"
/>

</div>
            <div>
              <h1 className="text-2xl font-bold">Lung Nodule Detection</h1>
            </div>
          </div>
          {results && (
            <div className="flex items-center gap-4 text-sm">
              <div className="bg-blue-500/20 px-4 py-2 rounded-lg border border-blue-400/30">
                <span className="text-blue-300">Instances: </span>
                <span className="font-bold">{results.num_instances}</span>
              </div>
              <div className="bg-purple-500/20 px-4 py-2 rounded-lg border border-purple-400/30">
                <span className="text-purple-300">Shape: </span>
                <span className="font-bold">{results.image_shape.join(' × ')}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-6">
        {/* Error Banner */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 flex items-center gap-3">
            <AlertCircle className="w-8 h-8 text-yellow-400" />

            <p className="flex-1 text-red-200">{error}</p>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">
              <X className="w-5 h-5" />
            </button>
          </div>
        )}

        {/* Upload Section */}
        <div className="bg-slate-900 border border-slate-700/40
 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-lg flex items-center justify-center">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold">Upload CT Scan</h2>
              <p className="text-sm text-blue-300">Upload both .mhd and .raw files</p>
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-4">
            {/* MHD File */}
            <div className="relative">
              <input
                type="file"
                accept=".mhd"
                onChange={(e) => handleFileUpload(e.target.files[0], "mhd")}
                className="hidden"
                id="mhd-upload"
                disabled={isUploading}
              />
              <label
                htmlFor="mhd-upload"
                className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                  uploadStatus.mhd
                    ? "border-green-500/50 bg-green-500/10"
                    : "border-slate-600 bg-slate-800/50 hover:bg-white/10 hover:border-white/40"
                }`}
              >
                {uploadStatus.mhd ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-green-400 mb-2" />
                    <span className="font-medium text-green-300">
                      {mhdFile?.name}
                    </span>
                  </>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-blue-400 mb-2" />
                    <span className="font-medium">Upload .mhd file</span>
                    <span className="text-xs text-gray-400 mt-1">Header file</span>
                  </>
                )}
              </label>
            </div>

            {/* RAW File */}
            <div className="relative">
              <input
                type="file"
                accept=".raw"
                onChange={(e) => handleFileUpload(e.target.files[0], "raw")}
                className="hidden"
                id="raw-upload"
                disabled={!scanId || isUploading}
              />
              <label
                htmlFor="raw-upload"
                className={`flex flex-col items-center justify-center p-6 border-2 border-dashed rounded-xl transition-all ${
                  !scanId
                    ? "border-gray-600 bg-gray-800/30 cursor-not-allowed opacity-50"
                    : uploadStatus.raw
                    ? "border-green-500/50 bg-green-500/10 cursor-pointer"
                    : "border-white/20 bg-white/5 hover:bg-white/10 hover:border-white/40 cursor-pointer"
                }`}
              >
                {uploadStatus.raw ? (
                  <>
                    <CheckCircle className="w-8 h-8 text-green-400 mb-2" />
                    <span className="font-medium text-green-300">
                      {rawFile?.name}
                    </span>
                  </>
                ) : (
                  <>
                    <Upload className="w-8 h-8 text-blue-400 mb-2" />
                    <span className="font-medium">Upload .raw file</span>
                    <span className="text-xs text-gray-400 mt-1">Image data</span>
                  </>
                )}
              </label>
            </div>
          </div>
        </div>

        {/* Configuration Section */}
        <div className="bbg-slate-900 border border-slate-700/40
 rounded-2xl p-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5" fill="white" viewBox="0 0 24 24">
                <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>
              </svg>
            </div>
            <div>
              <h2 className="text-xl font-bold">Analysis Configuration</h2>
              <p className="text-sm text-purple-300">Adjust detection parameters</p>
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-3">
                <label className="text-sm font-medium">Confidence Threshold</label>
                <span className="text-lg font-bold text-yellow-400">
  {(scoreThreshold * 100).toFixed(0)}%
</span>

              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={scoreThreshold}
                onChange={(e) => setScoreThreshold(Number(e.target.value))}
                className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                style={{
  background: `linear-gradient(
  to right,
  #22c55e 0%,
  #facc15 ${scoreThreshold * 100}%,
  rgba(255,255,255,0.1) ${scoreThreshold * 100}%
)`


                }}
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
              </div>
            </div>

            <div>
              <label className="text-sm font-medium block mb-3">Overlap Strategy</label>
              <div className="grid grid-cols-3 gap-3">
                {["highest_score", "first_wins", "last_wins"].map((strategy) => (
                  <button
                    key={strategy}
                    onClick={() => setOverlapStrategy(strategy)}
                    className={`px-4 py-3 rounded-lg font-medium text-sm transition-all ${
                      overlapStrategy === strategy
                        ? "bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg"
                        : "bg-white/5 hover:bg-white/10 text-gray-300"
                    }`}
                  >
                    {strategy.split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ══════════════════════════════════════════════
            EVALUATION: Ground Truth Section (NEW)
            ══════════════════════════════════════════════ */}
        <div className="bg-slate-900 border border-slate-700/40 rounded-2xl overflow-hidden">
          <button
            onClick={() => setShowEvalSection(!showEvalSection)}
            className="w-full p-6 flex items-center justify-between hover:bg-white/[0.02] transition-colors"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-lg flex items-center justify-center">
                <Activity className="w-5 h-5" />
              </div>
              <div className="text-left">
                <h2 className="text-xl font-bold">Evaluation Metrics</h2>
                <p className="text-sm text-amber-300">
                  {annotationsLoaded
                    ? `✓ Annotations loaded (${annotationsInfo?.num_nodules || 0} nodules across ${annotationsInfo?.num_series || 0} scans)`
                    : "Upload LUNA16 annotations.csv to enable evaluation"}
                </p>
              </div>
            </div>
            <svg
              className={`w-5 h-5 text-gray-400 transition-transform ${showEvalSection ? "rotate-180" : ""}`}
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {showEvalSection && (
            <div className="px-6 pb-6 space-y-4 border-t border-slate-700/40 pt-4">
              {/* Annotations Upload */}
              {!annotationsLoaded ? (
                <div>
                  <p className="text-sm text-gray-400 mb-3">
                    Upload the <code className="text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded text-xs">annotations.csv</code> file from LUNA16 dataset to compare detections against ground truth.
                  </p>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={(e) => handleAnnotationsUpload(e.target.files[0])}
                    className="hidden"
                    id="annotations-upload"
                  />
                  <label
                    htmlFor="annotations-upload"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-amber-500/15 hover:bg-amber-500/25 border border-amber-400/30 rounded-lg cursor-pointer transition-all text-amber-300 font-medium text-sm"
                  >
                    <Upload className="w-4 h-4" />
                    Upload annotations.csv
                  </label>
                </div>
              ) : (
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-5 h-5 text-green-400" />
                  <span className="text-green-300 text-sm font-medium">
                    Annotations loaded: {annotationsInfo?.num_nodules} nodules, {annotationsInfo?.num_series} scans
                  </span>
                </div>
              )}

              {/* Seriesuid field — auto-filled from .mhd filename, editable */}
              {annotationsLoaded && (
                <div>
                  <label className="text-sm font-medium text-gray-300 block mb-2">
                    Series UID
                    <span className="text-xs text-gray-500 ml-2">(auto-extracted from .mhd filename)</span>
                  </label>
                  <input
                    type="text"
                    value={seriesuid}
                    onChange={(e) => setSeriesuid(e.target.value)}
                    placeholder="1.3.6.1.4.1.14519.5.2.1..."
                    className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2.5 text-sm font-mono text-gray-200 placeholder-gray-500 focus:outline-none focus:border-amber-400/50 focus:ring-1 focus:ring-amber-400/20 transition-all"
                  />
                </div>
              )}

              {/* Run Evaluation Button */}
              {annotationsLoaded && results && (
                <button
                  onClick={runEvaluation}
                  disabled={isEvaluating || !seriesuid}
                  className="w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white py-3 rounded-xl font-bold text-base flex items-center justify-center gap-3 transition-all shadow-lg"
                >
                  {isEvaluating ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Evaluating against ground truth...
                    </>
                  ) : (
                    <>
                      <Activity className="w-5 h-5" />
                      Run Evaluation
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Run Analysis Button */}
        <button
          onClick={runInference}
          disabled={isInferring || !uploadStatus.mhd || !uploadStatus.raw}
          className="w-full bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all shadow-lg hover:shadow-xl disabled:shadow-none"
        >
          {isInferring ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Play className="w-5 h-5" />
              Run Analysis
            </>
          )}
        </button>

        {/* Results Section */}
        {results && results.num_instances > 0 && (
          <>
            {/* Statistics Cards */}
            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 border border-blue-400/30 rounded-xl p-4">
                <div className="text-sm text-blue-300 mb-1">Total Nodules</div>
                <div className="text-3xl font-bold">{results.num_instances}</div>
              </div>
              <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 border border-green-400/30 rounded-xl p-4">
                <div className="text-sm text-green-300 mb-1">Avg Volume</div>
                <div className="text-3xl font-bold">{results.stats.avg_instance_volume_mm3?.toFixed(0) || 0}</div>
                <div className="text-xs text-green-400">mm³</div>
              </div>
              <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 border border-purple-400/30 rounded-xl p-4">
                <div className="text-sm text-purple-300 mb-1">High Confidence</div>
                <div className="text-3xl font-bold">{results.stats.high_conf_instances || 0}</div>
              </div>
              <div className="bg-gradient-to-br from-yellow-400/20 to-amber-500/20 border border-yellow-400/30 rounded-xl p-4">

                <div className="text-sm text-orange-300 mb-1">Avg Confidence</div>
                <div className="text-3xl font-bold">{(results.stats.mean_confidence * 100).toFixed(0)}%</div>
              </div>
            </div>

            {/* ══════════════════════════════════════════════
                EVALUATION METRICS DASHBOARD (NEW)
                ══════════════════════════════════════════════ */}
            {evalResults && (
              <div className="bg-slate-900 border border-amber-500/20 rounded-2xl p-6 space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gradient-to-br from-amber-500 to-orange-500 rounded-lg flex items-center justify-center">
                      <Activity className="w-5 h-5" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold">Evaluation Results</h2>
                      <p className="text-sm text-amber-300">
                        GT: {evalResults.num_gt} nodules — Pred: {evalResults.num_pred} detections
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 bg-green-500/20 border border-green-400/30 rounded-full text-xs font-bold text-green-300">
                      TP {evalResults.true_positives}
                    </span>
                    <span className="px-3 py-1 bg-red-500/20 border border-red-400/30 rounded-full text-xs font-bold text-red-300">
                      FP {evalResults.false_positives}
                    </span>
                    <span className="px-3 py-1 bg-yellow-500/20 border border-yellow-400/30 rounded-full text-xs font-bold text-yellow-300">
                      FN {evalResults.false_negatives}
                    </span>
                  </div>
                </div>

                {/* ── Detection Metrics Row ── */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
                    Detection Metrics
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {/* Sensitivity */}
                    <div className={`bg-gradient-to-br ${metricCardBg(evalResults.sensitivity)} border rounded-xl p-4`}>
                      <div className="text-xs text-gray-400 mb-1">Sensitivity (Recall)</div>
                      <div className={`text-2xl font-bold ${metricColor(evalResults.sensitivity)}`}>
                        {fmtPct(evalResults.sensitivity)}
                      </div>
                      <div className="mt-2 w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-green-500 to-emerald-400 rounded-full transition-all"
                          style={{ width: `${(evalResults.sensitivity || 0) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Precision */}
                    <div className={`bg-gradient-to-br ${metricCardBg(evalResults.precision)} border rounded-xl p-4`}>
                      <div className="text-xs text-gray-400 mb-1">Precision</div>
                      <div className={`text-2xl font-bold ${metricColor(evalResults.precision)}`}>
                        {fmtPct(evalResults.precision)}
                      </div>
                      <div className="mt-2 w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-400 rounded-full transition-all"
                          style={{ width: `${(evalResults.precision || 0) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* F1 */}
                    <div className={`bg-gradient-to-br ${metricCardBg(evalResults.f1_score)} border rounded-xl p-4`}>
                      <div className="text-xs text-gray-400 mb-1">F1 Score</div>
                      <div className={`text-2xl font-bold ${metricColor(evalResults.f1_score)}`}>
                        {fmtPct(evalResults.f1_score)}
                      </div>
                      <div className="mt-2 w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-purple-500 to-pink-400 rounded-full transition-all"
                          style={{ width: `${(evalResults.f1_score || 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* ── Segmentation Metrics Row ── */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M3 3h18v18H3V3zm16 16V5H5v14h14z"/></svg>
                    Segmentation Metrics
                  </h3>
                  <div className="grid grid-cols-2 gap-3">
                    {/* Volumetric Dice */}
                    <div className={`bg-gradient-to-br ${metricCardBg(evalResults.mean_dice)} border rounded-xl p-4`}>
                      <div className="text-xs text-gray-400 mb-1">Volumetric Dice</div>
                      <div className={`text-2xl font-bold ${metricColor(evalResults.mean_dice)}`}>
                        {fmtMetric(evalResults.mean_dice)}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">GT sphere vs pred box mask</div>
                      <div className="mt-2 w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-teal-500 to-cyan-400 rounded-full transition-all"
                          style={{ width: `${(evalResults.mean_dice || 0) * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Volumetric IoU */}
                    <div className={`bg-gradient-to-br ${metricCardBg(evalResults.mean_iou)} border rounded-xl p-4`}>
                      <div className="text-xs text-gray-400 mb-1">Volumetric IoU</div>
                      <div className={`text-2xl font-bold ${metricColor(evalResults.mean_iou)}`}>
                        {fmtMetric(evalResults.mean_iou)}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">Stricter than Dice</div>
                      <div className="mt-2 w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-500 to-violet-400 rounded-full transition-all"
                          style={{ width: `${(evalResults.mean_iou || 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>


                {/* ── Per-Nodule Match Details ── */}
                {evalResults.matches && evalResults.matches.length > 0 && (
                  <div>
                    <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M4 6h18V4H4c-1.1 0-2 .9-2 2v11H0v3h14v-3H4V6zm19 2h-6c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h6c.55 0 1-.45 1-1V9c0-.55-.45-1-1-1zm-1 9h-4v-7h4v7z"/></svg>
                      Per-Nodule Match Details
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-white/10">
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">GT #</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Diameter</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Match</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Distance</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Box IoU</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Dice</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Mask IoU</th>
                            <th className="text-left py-2.5 px-3 text-xs font-medium text-gray-400">Score</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evalResults.matches.map((m, idx) => (
                            <tr key={idx} className="border-b border-white/5 hover:bg-white/[0.03]">
                              <td className="py-2.5 px-3 font-mono font-bold text-amber-400">#{m.gt_idx + 1}</td>
                              <td className="py-2.5 px-3">
                                {m.gt_diameter_mm?.toFixed(1)} <span className="text-gray-500">mm</span>
                              </td>
                              <td className="py-2.5 px-3">
                                {m.matched ? (
                                  <span className="px-2 py-0.5 bg-green-500/20 border border-green-400/30 rounded text-xs font-bold text-green-300">TP</span>
                                ) : (
                                  <span className="px-2 py-0.5 bg-red-500/20 border border-red-400/30 rounded text-xs font-bold text-red-300">FN</span>
                                )}
                              </td>
                              <td className="py-2.5 px-3 font-mono text-xs">
                                {m.matched ? `${m.distance_mm} mm` : "—"}
                              </td>
                              <td className={`py-2.5 px-3 font-mono text-xs ${metricColor(m.box_iou_3d)}`}>
                                {m.matched ? fmtMetric(m.box_iou_3d) : "—"}
                              </td>
                              <td className={`py-2.5 px-3 font-mono text-xs ${metricColor(m.instance_dice)}`}>
                                {fmtMetric(m.instance_dice)}
                              </td>
                              <td className={`py-2.5 px-3 font-mono text-xs ${metricColor(m.instance_iou)}`}>
                                {fmtMetric(m.instance_iou)}
                              </td>
                              <td className="py-2.5 px-3 font-mono text-xs">
                                {m.pred_score !== null ? fmtPct(m.pred_score) : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Info footer */}
                <div className="text-xs text-gray-500 border-t border-white/5 pt-3 flex items-center gap-2">
                  <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg>
                  GT masks are spheres generated from LUNA16 center + diameter annotations. Pred masks are axis-aligned boxes from pseudo-panoptic.
                  FROC/CPM requires running evaluation across multiple scans via <code className="text-amber-400/60 mx-1">run_evaluation.py</code>.
                </div>
              </div>
            )}

            {/* Nodule List */}
            <div className="bg-slate-900 border border-slate-700/40
 rounded-2xl p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                <Box className="w-5 h-5" />
                Detected Nodules
              </h2>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-white/10">
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">ID</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Diameter</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Volume</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Confidence</th>
                      <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.details.map((n) => (
                      <tr
                        key={n.instance_id}
                        onClick={() => {
  setSelectedInstanceId(n.instance_id);
  if (n.centroid) {
    setSelectedSlice(getSliceFromCentroid(n.centroid));
  }
}}
                        className={`border-b border-white/5 cursor-pointer transition-colors ${
                         selectedInstanceId === n.instance_id
  ? "bg-slate-800 ring-1 ring-yellow-400/40"
  : "hover:bg-white/5"

                        }`}
                      >
                        <td className="py-3 px-4">
                          <span className="font-mono font-bold text-blue-400">#{n.instance_id}</span>
                        </td>
                        <td className="py-3 px-4">
                          <span className="font-medium">{n.diameter_mm?.toFixed(1) || 'N/A'}</span>
                          <span className="text-xs text-gray-400 ml-1">mm</span>
                        </td>
                        <td className="py-3 px-4">
                          <span className="font-medium">{n.volume_mm3?.toFixed(0) || 'N/A'}</span>
                          <span className="text-xs text-gray-400 ml-1">mm³</span>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-2 bg-white/10 rounded-full overflow-hidden">
                              <div
                                className={`h-full bg-gradient-to-r ${getConfidenceColor(n.confidence)}`}
                                style={{ width: `${n.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-bold">{(n.confidence * 100).toFixed(0)}%</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <span
  className={`px-3 py-1 rounded-full text-xs font-medium
    ${n.confidence >= 0.7
      ? "bg-green-500 text-black"
      : n.confidence >= 0.5
      ? "bg-yellow-400 text-black"
      : "bg-red-500 text-white"}
  `}
>

  {getRiskLevel(n.confidence)}
</span>

                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Visualization Section */}
            <div className="bg-slate-900 border border-slate-700/40
 rounded-2xl p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                    <Eye className="w-5 h-5" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Visualization</h2>
                    <p className="text-sm text-indigo-300">Interactive slice viewer</p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => setShowMontage(!showMontage)}
                    className="px-4 py-2 bg-indigo-500/20 hover:bg-indigo-500/30 border border-indigo-400/30 rounded-lg flex items-center gap-2 transition-all"
                  >
                    <Grid3x3 className="w-4 h-4" />
                    Montage
                  </button>
                  <button
                    onClick={() => setShow3D(!show3D)}
                    className="px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 border border-purple-400/30 rounded-lg flex items-center gap-2 transition-all"
                  >
                    <Box className="w-4 h-4" />
                    3D View
                  </button>
                </div>
              </div>

              {/* View Controls */}
              <div className="grid md:grid-cols-3 gap-4 mb-4">
                {["axial", "sagittal", "coronal"].map((view) => (
                  <button
                    key={view}
                    onClick={() => {
  setSelectedView(view);
  setSelectedSlice(() => {
  if (selectedInstanceId && results) {
    const inst = results.details.find(
      (d) => d.instance_id === selectedInstanceId
    );
    if (inst?.centroid) {
      return getSliceFromCentroid(inst.centroid);
    }
  }
  return Math.min(selectedSlice, getMaxSlice());
});
}}
                    className={`px-4 py-3 rounded-lg font-medium transition-all ${
                      selectedView === view
                        ? "bg-gradient-to-r from-indigo-500 to-purple-500 text-white shadow-lg"
                        : "bg-white/5 hover:bg-white/10 text-gray-300"
                    }`}
                  >
                    {view.charAt(0).toUpperCase() + view.slice(1)} View
                  </button>
                ))}
              </div>

              {/* Slice Slider */}
              <div className="mb-6">
                <div className="flex justify-between items-center mb-3">
                  <label className="text-sm font-medium">Slice Position</label>
                  <span className="text-lg font-bold text-indigo-400">
                    {selectedSlice} / {getMaxSlice()}
                  </span>
                </div>
                <input
  type="range"
  min="0"
  max={getMaxSlice()}
  value={selectedSlice}
  onChange={(e) => setSelectedSlice(Number(e.target.value))}
  className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
/>
              </div>

              {/* Image Display */}
              <div className="bg-black rounded-xl overflow-hidden">
                <img
  src={`${API_URL}/visualize/${scanId}/slice/${getSafeSlice()}?view=${selectedView}&t=${Date.now()}`}
  alt={`${selectedView} slice`}
  className="w-full"
  onError={(e) => {
    e.target.src = "";
  }}
/>

              </div>
            </div>

            {/* Montage Modal */}
            {showMontage && (
              <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6">
                <div className="bg-slate-900 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-auto">
                  <div className="sticky top-0 bg-slate-900 border-b border-white/10 p-4 flex justify-between items-center">
                    <h3 className="text-xl font-bold">Montage View - {selectedView}</h3>
                    <button onClick={() => setShowMontage(false)} className="p-2 hover:bg-white/10 rounded-lg">
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-6">
                    <img
                      src={`${API_URL}/visualize/${scanId}/montage?view=${selectedView}&t=${Date.now()}`}
                      alt="Montage view"
                      className="w-full"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* 3D Modal */}
            {show3D && (
              <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-6">
                <div className="bg-slate-900 rounded-2xl max-w-6xl w-full max-h-[90vh] overflow-auto">
                  <div className="sticky top-0 bg-slate-900 border-b border-white/10 p-4 flex justify-between items-center">
                    <h3 className="text-xl font-bold">3D Maximum Intensity Projection</h3>
                    <button onClick={() => setShow3D(false)} className="p-2 hover:bg-white/10 rounded-lg">
                      <X className="w-5 h-5" />
                    </button>
                  </div>
                  <div className="p-6">
                    <img
  src={build3DUrl()}
  alt="3D Maximum Intensity Projection"
  className="w-full"
/>


                  </div>
                </div>
              </div>
            )}
          </>
        )}

        {/* No Results Message */}
        {results && results.num_instances === 0 && (
          <div className="bg-slate-900 border border-slate-700/40
 rounded-2xl p-12 text-center">
            <div className="w-16 h-16 bg-yellow-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
              <AlertCircle className="w-8 h-8 text-yellow-400" />
            </div>
            <h3 className="text-xl font-bold mb-2">No Nodules Detected</h3>
            <p className="text-gray-400">Try adjusting the confidence threshold or upload a different scan</p>
          </div>
        )}

        {/* Reset Button */}
        {(results || scanId) && (
          <button
            onClick={cleanup}
            className="w-full bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 text-white py-4 rounded-xl font-bold text-lg flex items-center justify-center gap-3 transition-all shadow-lg hover:shadow-xl"
          >
            <RefreshCw className="w-5 h-5" />
            Start New Analysis
          </button>
        )}
      </div>
    </div>
  );
}
