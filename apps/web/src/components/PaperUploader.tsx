"use client";

import { useRef, useState } from "react";
import type { Paper } from "@/types";

interface PaperUploaderProps {
  projectId: string | null;
  apiUrl: string;
  papers: Paper[];
  selectedPaper: Paper | null;
  onPaperSelect: (paper: Paper) => void;
  onUploadSuccess: () => void;
}

export default function PaperUploader({
  projectId,
  apiUrl,
  papers,
  selectedPaper,
  onPaperSelect,
  onUploadSuccess,
}: PaperUploaderProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStage, setUploadStage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const pollJobStatus = async (jobId: string) => {
    let attempts = 0;
    const maxAttempts = 60;

    const interval = setInterval(async () => {
      attempts += 1;
      try {
        const res = await fetch(`${apiUrl}/api/v1/jobs/${jobId}`);
        if (!res.ok) throw new Error("Failed to check job status");
        const job = await res.json();

        setUploadStage(job.stage);
        setProgress(Math.round(job.progress * 100));

        if (job.status === "COMPLETED") {
          clearInterval(interval);
          setIsUploading(false);
          setUploadStage(null);
          onUploadSuccess();
        } else if (job.status === "FAILED") {
          clearInterval(interval);
          setIsUploading(false);
          setErrorMessage(job.error_message || "Ingestion pipeline failed");
          onUploadSuccess();
        } else if (attempts >= maxAttempts) {
          clearInterval(interval);
          setIsUploading(false);
          setErrorMessage("Ingestion timed out polling job");
        }
      } catch {
        clearInterval(interval);
        setIsUploading(false);
        setErrorMessage("Network error while checking job progress");
      }
    }, 1500);
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !projectId) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setErrorMessage("Only PDF documents are supported");
      return;
    }

    setIsUploading(true);
    setUploadStage("UPLOADING");
    setProgress(10);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${apiUrl}/api/v1/projects/${projectId}/papers`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(err.detail || "Upload failed");
      }

      const data = await res.json();
      pollJobStatus(data.job_id);
    } catch (err: unknown) {
      setIsUploading(false);
      setUploadStage(null);
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to upload PDF",
      );
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4 shadow-xs">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-zinc-900">
            Papers in Project
          </h2>
          <p className="text-xs text-zinc-500">
            Upload research PDFs to index and ask questions.
          </p>
        </div>

        <div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".pdf,application/pdf"
            className="hidden"
            disabled={!projectId || isUploading}
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={!projectId || isUploading}
            className="rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
          >
            {isUploading ? "Processing…" : "Upload PDF"}
          </button>
        </div>
      </div>

      {/* Uploading progress banner */}
      {isUploading && (
        <div className="mt-3 rounded-lg bg-zinc-50 border border-zinc-200 p-3">
          <div className="flex justify-between text-xs text-zinc-700 font-medium">
            <span>Stage: {uploadStage}</span>
            <span>{progress}%</span>
          </div>
          <div className="mt-1.5 h-1.5 w-full rounded-full bg-zinc-200 overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="mt-3 rounded-lg bg-rose-50 border border-rose-200 p-2.5 text-xs text-rose-800">
          {errorMessage}
        </div>
      )}

      {/* Papers list */}
      <div className="mt-3 flex flex-wrap gap-2">
        {papers.length === 0 ? (
          <p className="text-xs text-zinc-400 py-2">No papers uploaded yet.</p>
        ) : (
          papers.map((paper) => {
            const isSelected = selectedPaper?.id === paper.id;
            return (
              <button
                key={paper.id}
                type="button"
                onClick={() => onPaperSelect(paper)}
                className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                  isSelected
                    ? "border-zinc-900 bg-zinc-900 text-white"
                    : "border-zinc-200 bg-zinc-50 text-zinc-800 hover:bg-zinc-100"
                }`}
              >
                <span className="truncate max-w-[200px]">{paper.filename}</span>
                <span
                  className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                    paper.status === "READY"
                      ? isSelected
                        ? "bg-emerald-800 text-emerald-100"
                        : "bg-emerald-100 text-emerald-800"
                      : isSelected
                        ? "bg-amber-800 text-amber-100"
                        : "bg-amber-100 text-amber-800"
                  }`}
                >
                  {paper.status}
                </span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
