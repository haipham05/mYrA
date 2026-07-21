"use client";

import { useRef, useState } from "react";
import type { Paper } from "@/types";

interface PaperUploaderProps {
  projectId: string | null;
  apiUrl: string;
  papers: Paper[];
  total: number;
  offset: number;
  selectedPaper: Paper | null;
  onPaperSelect: (paper: Paper) => void;
  onUploadSuccess: () => void;
  onSearch: (filters: {
    q?: string;
    status?: string;
    year?: string;
    offset: number;
  }) => Promise<void>;
  paperScope: "paper" | "selection" | "project";
  selectedPaperIds: string[];
  onScopeChange: (scope: "paper" | "selection" | "project") => Promise<void>;
  onSelectedPaperIdsChange: (
    paperId: string,
    checked: boolean,
  ) => Promise<void>;
}

export default function PaperUploader({
  projectId,
  apiUrl,
  papers,
  total,
  offset,
  selectedPaper,
  onPaperSelect,
  onUploadSuccess,
  onSearch,
  paperScope,
  selectedPaperIds,
  onScopeChange,
  onSelectedPaperIdsChange,
}: PaperUploaderProps) {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStage, setUploadStage] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [yearFilter, setYearFilter] = useState("");
  const [searching, setSearching] = useState(false);
  const [scopeSaving, setScopeSaving] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const runSearch = async (nextOffset = 0) => {
    setSearching(true);
    setErrorMessage(null);
    try {
      await onSearch({
        q: query.trim() || undefined,
        status: statusFilter || undefined,
        year: yearFilter || undefined,
        offset: nextOffset,
      });
    } catch (error: unknown) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not search papers.",
      );
    } finally {
      setSearching(false);
    }
  };

  const updateScope = async (save: () => Promise<void>) => {
    setScopeSaving(true);
    setErrorMessage(null);
    try {
      await save();
    } catch (error: unknown) {
      setErrorMessage(
        error instanceof Error ? error.message : "Could not save chat scope.",
      );
    } finally {
      setScopeSaving(false);
    }
  };

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
          setQuery("");
          setStatusFilter("");
          setYearFilter("");
          onUploadSuccess();
        } else if (job.status === "FAILED") {
          clearInterval(interval);
          setIsUploading(false);
          setErrorMessage(job.error_message || "Ingestion pipeline failed");
          setQuery("");
          setStatusFilter("");
          setYearFilter("");
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

      <form
        className="mt-4 grid gap-2 rounded-lg bg-zinc-50 p-3 sm:grid-cols-[minmax(12rem,1fr)_9rem_7rem_auto]"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch(0);
        }}
      >
        <label className="sr-only" htmlFor="paper-search">
          Search papers
        </label>
        <input
          id="paper-search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          maxLength={200}
          placeholder="Search title, author, or filename"
          className="min-w-0 rounded border border-zinc-300 bg-white px-2.5 py-1.5 text-xs"
        />
        <label className="sr-only" htmlFor="paper-status-filter">
          Filter by status
        </label>
        <select
          id="paper-status-filter"
          value={statusFilter}
          onChange={(event) => setStatusFilter(event.target.value)}
          className="rounded border border-zinc-300 bg-white px-2 py-1.5 text-xs"
        >
          <option value="">Any status</option>
          <option value="READY">Ready</option>
          <option value="PROCESSING">Processing</option>
          <option value="FAILED">Failed</option>
        </select>
        <label className="sr-only" htmlFor="paper-year-filter">
          Filter by year
        </label>
        <input
          id="paper-year-filter"
          type="number"
          min={1000}
          max={2100}
          value={yearFilter}
          onChange={(event) => setYearFilter(event.target.value)}
          placeholder="Year"
          className="rounded border border-zinc-300 bg-white px-2 py-1.5 text-xs"
        />
        <button
          type="submit"
          disabled={searching}
          className="rounded border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium disabled:opacity-50"
        >
          {searching ? "Searching…" : "Search"}
        </button>
      </form>

      <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
        <label htmlFor="paper-scope" className="font-medium text-zinc-700">
          Chat searches
        </label>
        <select
          id="paper-scope"
          value={paperScope}
          disabled={!projectId || scopeSaving}
          onChange={(event) =>
            void updateScope(() =>
              onScopeChange(
                event.target.value as "paper" | "selection" | "project",
              ),
            )
          }
          className="rounded border border-zinc-300 bg-white px-2 py-1.5"
        >
          <option value="project">Whole project</option>
          <option value="paper">Current paper</option>
          <option value="selection">Selected papers</option>
        </select>
        <span className="text-zinc-500">
          {paperScope === "project"
            ? "All project papers are in scope."
            : `${selectedPaperIds.length} paper${selectedPaperIds.length === 1 ? "" : "s"} selected.`}
        </span>
      </div>

      {/* Papers list */}
      <div className="mt-3 flex flex-wrap gap-2">
        {papers.length === 0 ? (
          <p className="text-xs text-zinc-400 py-2">No papers uploaded yet.</p>
        ) : (
          papers.map((paper) => {
            const isSelected = selectedPaper?.id === paper.id;
            return (
              <div
                key={paper.id}
                className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-zinc-50 pr-2"
              >
                <button
                  type="button"
                  onClick={() => onPaperSelect(paper)}
                  className={`flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer ${
                    isSelected
                      ? "bg-zinc-900 text-white"
                      : "text-zinc-800 hover:bg-zinc-100"
                  }`}
                >
                  <span className="truncate max-w-[240px]">
                    {paper.title || paper.filename}
                  </span>
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
                <label className="flex items-center gap-1 text-[10px] text-zinc-500">
                  <input
                    type="checkbox"
                    aria-label={`Include ${paper.title || paper.filename} in chat scope`}
                    checked={selectedPaperIds.includes(paper.id)}
                    disabled={paperScope === "project" || scopeSaving}
                    onChange={(event) =>
                      void updateScope(() =>
                        onSelectedPaperIdsChange(
                          paper.id,
                          event.target.checked,
                        ),
                      )
                    }
                  />
                  Scope
                </label>
              </div>
            );
          })
        )}
      </div>
      <div className="mt-3 flex items-center justify-between text-xs text-zinc-500">
        <span>
          {total === 0
            ? "0 papers"
            : `Showing ${offset + 1}–${Math.min(offset + papers.length, total)} of ${total}`}
        </span>
        <div className="flex gap-2">
          <button
            type="button"
            disabled={offset <= 0 || searching}
            onClick={() => void runSearch(Math.max(0, offset - 50))}
            className="rounded border border-zinc-300 px-2 py-1 disabled:opacity-40"
          >
            Previous
          </button>
          <button
            type="button"
            disabled={offset + papers.length >= total || searching}
            onClick={() => void runSearch(offset + 50)}
            className="rounded border border-zinc-300 px-2 py-1 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
