"use client";

import { useEffect, useRef, useState } from "react";
import type { BoundingBox, Citation, Paper } from "@/types";

interface PdfViewerProps {
  paper: Paper | null;
  activeCitation: Citation | null;
  apiUrl: string;
}

export default function PdfViewer({
  paper,
  activeCitation,
  apiUrl,
}: PdfViewerProps) {
  const [userPage, setUserPage] = useState<number | null>(null);
  const [scale, setScale] = useState<number>(1.2);
  const [rotation, setRotation] = useState<number>(0);
  const [prevCitation, setPrevCitation] = useState<Citation | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);

  // Sync active citation page
  if (activeCitation !== prevCitation) {
    setPrevCitation(activeCitation);
    setUserPage(null);
  }

  const citationPage =
    activeCitation && activeCitation.paper_id === paper?.id
      ? activeCitation.page_number
      : 1;
  const currentPage = userPage ?? citationPage;

  // Handle PDF document rendering with PDF.js if available in browser
  useEffect(() => {
    let cancelled = false;

    async function renderPage() {
      if (!paper) return;

      try {
        // Dynamic import to avoid SSR issues
        const pdfjsLib = await import("pdfjs-dist");
        if (
          pdfjsLib.GlobalWorkerOptions &&
          !pdfjsLib.GlobalWorkerOptions.workerSrc
        ) {
          pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;
        }

        const docUrl = `${apiUrl}/api/v1/papers/${paper.id}/document`;
        const loadingTask = pdfjsLib.getDocument({ url: docUrl });
        const pdfDoc = await loadingTask.promise;

        if (cancelled) return;

        const page = await pdfDoc.getPage(currentPage);
        if (cancelled) return;

        const viewport = page.getViewport({ scale, rotation });
        const canvas = canvasRef.current;
        if (!canvas) return;

        const context = canvas.getContext("2d");
        if (!context) return;

        canvas.height = viewport.height;
        canvas.width = viewport.width;

        const renderContext = {
          canvasContext: context,
          viewport,
        };

        await page.render(renderContext).promise;

        // Render text layer for text matching
        const textContent = await page.getTextContent();
        if (cancelled) return;

        const textLayerDiv = textLayerRef.current;
        if (textLayerDiv) {
          textLayerDiv.innerHTML = "";
          textLayerDiv.style.width = `${viewport.width}px`;
          textLayerDiv.style.height = `${viewport.height}px`;

          // Populate text spans
          for (const item of textContent.items) {
            if ("str" in item) {
              const span = document.createElement("span");
              span.textContent = item.str;
              textLayerDiv.appendChild(span);
            }
          }
        }
      } catch {
        // Fallback gracefully in testing / non-canvas environments
      }
    }

    renderPage();

    return () => {
      cancelled = true;
    };
  }, [paper, currentPage, scale, rotation, apiUrl]);

  // Derive highlight status purely during render
  let highlightStatus: "exact" | "approximate" | "unavailable" = "unavailable";
  if (activeCitation && activeCitation.paper_id === paper?.id) {
    if (activeCitation.anchor_status === "unresolved") {
      highlightStatus = "unavailable";
    } else if (
      activeCitation.anchor_status === "verified" ||
      (activeCitation.anchors && activeCitation.anchors.length > 0)
    ) {
      highlightStatus = "exact";
    } else if (activeCitation.bounding_boxes.length > 0) {
      highlightStatus = "approximate";
    } else {
      highlightStatus = "unavailable";
    }
  }

  if (!paper) {
    return (
      <div className="flex h-full min-h-[400px] flex-col items-center justify-center rounded-xl border border-dashed border-zinc-300 p-8 text-center text-zinc-500">
        <p className="text-base font-medium">No paper selected</p>
        <p className="mt-1 text-sm">
          Upload or select a paper to view its document and citations.
        </p>
      </div>
    );
  }

  const isCurrentPageCited =
    activeCitation &&
    activeCitation.paper_id === paper.id &&
    activeCitation.page_number === currentPage;

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xs">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-zinc-200 px-4 py-3">
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-sm font-semibold text-zinc-900">
            {paper.filename}
          </h2>
          <div className="flex items-center gap-2 text-xs text-zinc-500">
            <span>Status: {paper.status}</span>
            {paper.page_count && <span>• {paper.page_count} pages</span>}
          </div>
        </div>

        {/* Viewport & Page Controls */}
        <div className="flex items-center gap-3">
          {/* Zoom controls */}
          <div className="flex items-center gap-1 rounded-md border border-zinc-200 bg-zinc-50 px-1 py-0.5 text-xs">
            <button
              type="button"
              onClick={() => setScale((s) => Math.max(0.6, s - 0.2))}
              className="px-1.5 py-0.5 hover:bg-zinc-200 rounded text-zinc-700 font-bold"
              title="Zoom Out"
            >
              -
            </button>
            <span className="px-1 text-zinc-600 font-mono text-[11px]">
              {Math.round(scale * 100)}%
            </span>
            <button
              type="button"
              onClick={() => setScale((s) => Math.min(2.5, s + 0.2))}
              className="px-1.5 py-0.5 hover:bg-zinc-200 rounded text-zinc-700 font-bold"
              title="Zoom In"
            >
              +
            </button>
            <button
              type="button"
              onClick={() => setRotation((r) => (r + 90) % 360)}
              className="ml-1 px-1.5 py-0.5 hover:bg-zinc-200 rounded text-zinc-600 text-[11px]"
              title="Rotate 90°"
            >
              ↻
            </button>
          </div>

          {/* Page navigation */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setUserPage(Math.max(1, currentPage - 1))}
              disabled={currentPage <= 1}
              className="rounded px-2 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40"
            >
              Previous
            </button>
            <span className="text-xs text-zinc-600 font-medium">
              Page {currentPage} of {paper.page_count || "?"}
            </span>
            <button
              type="button"
              onClick={() =>
                setUserPage(
                  paper.page_count
                    ? Math.min(paper.page_count, currentPage + 1)
                    : currentPage + 1,
                )
              }
              disabled={
                paper.page_count ? currentPage >= paper.page_count : false
              }
              className="rounded px-2 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Active Citation Callout */}
      {activeCitation && activeCitation.paper_id === paper.id && (
        <div className="border-b border-amber-200 bg-amber-50/80 px-4 py-2.5">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2">
              <span className="inline-flex items-center justify-center rounded bg-amber-200 px-1.5 py-0.5 text-xs font-semibold text-amber-900">
                [{activeCitation.citation_index}]
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-amber-900">
                  Cited on Page {activeCitation.page_number}
                </p>
                <p className="mt-0.5 text-xs text-amber-800 line-clamp-2 italic">
                  &ldquo;{activeCitation.quote}&rdquo;
                </p>
              </div>
            </div>
            {/* Status indicator */}
            {isCurrentPageCited && (
              <span
                className={`text-[11px] font-medium px-2 py-0.5 rounded-full shrink-0 ${
                  highlightStatus === "unavailable"
                    ? "bg-zinc-200 text-zinc-700"
                    : "bg-amber-100 text-amber-800 border border-amber-300"
                }`}
              >
                {highlightStatus === "unavailable"
                  ? "Exact highlight unavailable"
                  : highlightStatus === "exact"
                    ? "Verbatim match"
                    : "Evidence span"}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Main viewer viewport */}
      <div
        ref={containerRef}
        className="relative flex-1 overflow-auto bg-zinc-100 p-4 flex justify-center items-start"
      >
        <div className="relative shadow-md bg-white border border-zinc-200">
          {/* PDF canvas layer */}
          <canvas ref={canvasRef} className="block max-w-full" />

          {/* Selectable transparent text layer */}
          <div
            ref={textLayerRef}
            className="absolute inset-0 select-text overflow-hidden opacity-0 pointer-events-auto"
          />

          {/* High-precision overlay highlights for cited page */}
          {isCurrentPageCited &&
            highlightStatus !== "unavailable" &&
            activeCitation.bounding_boxes.map(
              (box: BoundingBox, idx: number) => {
                const isBottomLeft = box.origin === "BOTTOM_LEFT";
                const leftPct = (box.x_min / (box.page_width || 612)) * 100;
                const widthPct =
                  ((box.x_max - box.x_min || 100) / (box.page_width || 612)) *
                  100;
                const topPct = isBottomLeft
                  ? (((box.page_height || 792) - box.y_max) /
                      (box.page_height || 792)) *
                    100
                  : (box.y_min / (box.page_height || 792)) * 100;
                const heightPct =
                  ((box.y_max - box.y_min || 20) / (box.page_height || 792)) *
                  100;

                return (
                  <div
                    key={`bbox-${idx}`}
                    data-testid="evidence-highlight"
                    style={{
                      position: "absolute",
                      left: `${Math.max(0, leftPct)}%`,
                      top: `${Math.max(0, topPct)}%`,
                      width: `${Math.min(100, widthPct)}%`,
                      height: `${Math.min(100, heightPct)}%`,
                      backgroundColor: "rgba(245, 158, 11, 0.35)",
                      border: "2px solid #d97706",
                      boxShadow: "0 0 6px rgba(245, 158, 11, 0.4)",
                      borderRadius: "2px",
                      pointerEvents: "none",
                      zIndex: 20,
                    }}
                  />
                );
              },
            )}
        </div>
      </div>
    </div>
  );
}
