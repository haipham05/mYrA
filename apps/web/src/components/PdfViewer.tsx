"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import type { Citation, Paper, SourceSelection } from "@/types";

interface PdfViewerProps {
  paper: Paper | null;
  activeCitation: Citation | null;
  apiUrl: string;
  onExplainSelection?: (selection: SourceSelection | null) => void;
}

export interface HighlightRect {
  left: number;
  top: number;
  width: number;
  height: number;
}

/**
 * Searches the DOM text nodes of a container (e.g. PDF.js text layer) for the target quote,
 * supporting exact and whitespace-normalized substring matching, and returns a DOM Range.
 */
export function findRangeForQuote(
  container: HTMLElement,
  targetQuote: string,
): Range | null {
  if (!targetQuote || !targetQuote.trim()) return null;

  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const textNodes: { node: Text; start: number; end: number }[] = [];
  let fullText = "";
  let currentNode: Node | null;

  while ((currentNode = walker.nextNode())) {
    const textNode = currentNode as Text;
    const text = textNode.nodeValue || "";
    if (text.length > 0) {
      if (fullText.length > 0 && !/\s$/.test(fullText) && !/^\s/.test(text)) {
        fullText += " ";
      }
      textNodes.push({
        node: textNode,
        start: fullText.length,
        end: fullText.length + text.length,
      });
      fullText += text;
    }
  }

  if (textNodes.length === 0 || !fullText) return null;

  const candidateSpans: { matchStart: number; matchEnd: number }[] = [];

  // 1. Direct search for all occurrences
  let searchIdx = fullText.indexOf(targetQuote);
  while (searchIdx !== -1) {
    candidateSpans.push({
      matchStart: searchIdx,
      matchEnd: searchIdx + targetQuote.length,
    });
    searchIdx = fullText.indexOf(targetQuote, searchIdx + 1);
  }

  // 2. Whitespace-normalized search if direct search yielded no results
  if (candidateSpans.length === 0) {
    const rawToNormMap: number[] = [];
    let normFullText = "";
    let inWhitespace = false;

    for (let i = 0; i < fullText.length; i++) {
      const ch = fullText[i];
      if (/\s/.test(ch)) {
        if (!inWhitespace) {
          rawToNormMap.push(i);
          normFullText += " ";
          inWhitespace = true;
        }
      } else {
        rawToNormMap.push(i);
        normFullText += ch;
        inWhitespace = false;
      }
    }

    const normQuote = targetQuote.trim().replace(/\s+/g, " ");
    let normIndex = normFullText.indexOf(normQuote);

    while (normIndex !== -1) {
      const matchStart = rawToNormMap[normIndex];
      const normEndIndex = normIndex + normQuote.length - 1;
      const matchEnd =
        (rawToNormMap[normEndIndex] ?? matchStart + normQuote.length) + 1;
      candidateSpans.push({ matchStart, matchEnd });
      normIndex = normFullText.indexOf(normQuote, normIndex + 1);
    }
  }

  // Parser offsets and PDF.js text-item offsets are not the same coordinate
  // system. A duplicate phrase has no safe occurrence identity yet.
  if (candidateSpans.length !== 1) return null;
  const { matchStart, matchEnd } = candidateSpans[0];

  const startEntry =
    textNodes.find((tn) => matchStart >= tn.start && matchStart < tn.end) ||
    textNodes.find((tn) => matchStart <= tn.end);
  const endEntry =
    textNodes.find((tn) => matchEnd > tn.start && matchEnd <= tn.end) ||
    [...textNodes].reverse().find((tn) => matchEnd >= tn.start);

  if (!startEntry || !endEntry) return null;

  try {
    const range = document.createRange();
    const sOffset = Math.max(
      0,
      Math.min(startEntry.node.length, matchStart - startEntry.start),
    );
    const eOffset = Math.max(
      0,
      Math.min(endEntry.node.length, matchEnd - endEntry.start),
    );
    range.setStart(startEntry.node, sOffset);
    range.setEnd(endEntry.node, eOffset);
    return range;
  } catch {
    return null;
  }
}

export default function PdfViewer({
  paper,
  activeCitation,
  apiUrl,
  onExplainSelection,
}: PdfViewerProps) {
  const [userPage, setUserPage] = useState<number | null>(null);
  const [scale, setScale] = useState<number>(1.2);
  const [rotation, setRotation] = useState<number>(0);
  const [prevCitation, setPrevCitation] = useState<Citation | null>(null);

  const [highlightRects, setHighlightRects] = useState<HighlightRect[]>([]);
  const [isExactMatch, setIsExactMatch] = useState<boolean>(false);
  const [textLayerReady, setTextLayerReady] = useState<number>(0);
  const [textLayerRenderKey, setTextLayerRenderKey] = useState<string | null>(
    null,
  );

  const [renderError, setRenderError] = useState<string | null>(null);
  const [servedDocument, setServedDocument] = useState<{
    paperId: string;
    sha256: string;
  } | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const textLayerRef = useRef<HTMLDivElement>(null);
  const documentRef = useRef<{
    key: string;
    promise: Promise<{ pdfDoc: PDFDocumentProxy; sha256: string }>;
  } | null>(null);

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
  const paperId = paper?.id;
  const paperHash = paper?.document_sha256;
  const renderKey = `${paperId ?? "none"}:${currentPage}:${scale}:${rotation}`;

  const captureTextSelection = useCallback(() => {
    const selection = window.getSelection();
    const selectedText = selection?.toString().trim() ?? "";
    const range = selection?.rangeCount ? selection.getRangeAt(0) : null;
    if (
      !paperId ||
      !paperHash ||
      !range ||
      !textLayerRef.current?.contains(range.commonAncestorContainer)
    ) {
      return;
    }
    onExplainSelection?.(
      selectedText.length > 0 && selectedText.length <= 2000
        ? {
            paper_id: paperId,
            page_number: currentPage,
            quote: selectedText,
            document_sha256: paperHash,
          }
        : null,
    );
  }, [currentPage, onExplainSelection, paperHash, paperId]);

  // Handle PDF document rendering with PDF.js
  useEffect(() => {
    let cancelled = false;

    async function renderPage() {
      if (!paper) return;
      setRenderError(null);

      try {
        const pdfjsLib = await import("pdfjs-dist");
        if (pdfjsLib.GlobalWorkerOptions) {
          pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        }

        const docUrl = `${apiUrl}/api/v1/papers/${paper.id}/document`;
        const documentKey = `${docUrl}:${paper.document_sha256 ?? "unversioned"}`;
        if (documentRef.current?.key !== documentKey) {
          const promise = (async () => {
            const response = await fetch(docUrl, { cache: "no-store" });
            if (!response.ok) throw new Error("Failed to load the source PDF");
            const data = await response.arrayBuffer();
            const digest = await crypto.subtle.digest("SHA-256", data);
            const sha256 = Array.from(new Uint8Array(digest), (byte) =>
              byte.toString(16).padStart(2, "0"),
            ).join("");
            const pdfDoc = await pdfjsLib.getDocument({
              data: new Uint8Array(data),
            }).promise;
            return { pdfDoc, sha256 };
          })();
          documentRef.current = { key: documentKey, promise };
        }
        const { pdfDoc, sha256 } = await documentRef.current.promise;

        if (cancelled) return;
        setServedDocument({ paperId: paper.id, sha256 });

        const page = await pdfDoc.getPage(currentPage);
        if (cancelled) return;

        const viewport = page.getViewport({ scale, rotation });
        const canvas = canvasRef.current;
        if (!canvas) return;

        let context: CanvasRenderingContext2D | null = null;
        try {
          context = canvas.getContext("2d");
        } catch {
          // JSDOM does not implement getContext
        }

        if (context) {
          canvas.height = viewport.height;
          canvas.width = viewport.width;

          const renderContext = {
            canvasContext: context,
            viewport,
          };

          await page.render(renderContext).promise;
        }

        // Render text layer for text matching
        const textContent = await page.getTextContent();
        if (cancelled) return;

        const textLayerDiv = textLayerRef.current;
        if (textLayerDiv) {
          textLayerDiv.innerHTML = "";
          textLayerDiv.style.width = `${viewport.width}px`;
          textLayerDiv.style.height = `${viewport.height}px`;
          textLayerDiv.style.setProperty(
            "--scale-factor",
            String(viewport.scale),
          );

          if (pdfjsLib.TextLayer) {
            const textLayer = new pdfjsLib.TextLayer({
              textContentSource: textContent,
              container: textLayerDiv,
              viewport,
            });
            await textLayer.render();
          } else {
            textLayerDiv.innerHTML = "";
          }
          if (!cancelled) {
            setTextLayerRenderKey(renderKey);
            setTextLayerReady((c) => c + 1);
          }
        }
      } catch (err) {
        if (!cancelled) {
          documentRef.current = null;
          setRenderError(
            err instanceof Error ? err.message : "Failed to render PDF page",
          );
        }
      }
    }

    renderPage();

    return () => {
      cancelled = true;
    };
  }, [paper, currentPage, scale, rotation, apiUrl, renderKey]);

  useEffect(() => {
    const key = paperId
      ? `${apiUrl}/api/v1/papers/${paperId}/document:${paperHash ?? "unversioned"}`
      : null;
    return () => {
      const cached = documentRef.current;
      if (cached && cached.key === key) {
        documentRef.current = null;
        void cached.promise
          .then(({ pdfDoc }) => pdfDoc.destroy())
          .catch(() => {});
      }
    };
  }, [paperId, paperHash, apiUrl]);

  // Compute exact character highlights using DOM Range.getClientRects()
  const recomputeHighlights = useCallback(() => {
    const isCurrentPageCited = Boolean(
      activeCitation &&
      activeCitation.paper_id === paper?.id &&
      activeCitation.page_number === currentPage,
    );

    const matchingAnchors = activeCitation?.anchors?.filter(
      (a) =>
        a.page_number === currentPage &&
        a.exact_quote === activeCitation.quote &&
        a.anchor_status === "verified" &&
        a.document_sha256 === activeCitation.document_sha256 &&
        a.parser_version === activeCitation.parser_version &&
        (a.source_element_id ||
          a.parser_version === "translation-page-rawtext-v1") &&
        a.source_char_start != null &&
        a.source_char_end != null &&
        a.source_char_end > a.source_char_start,
    );
    const hasOneMatchingAnchor = matchingAnchors?.length === 1;
    const isHashMatched = Boolean(
      paper?.document_sha256 &&
      activeCitation?.document_sha256 &&
      paper.document_sha256 === activeCitation.document_sha256 &&
      servedDocument?.paperId === paper.id &&
      servedDocument.sha256 === paper.document_sha256,
    );

    if (
      !isCurrentPageCited ||
      !activeCitation ||
      activeCitation.anchor_status !== "verified" ||
      !hasOneMatchingAnchor ||
      !isHashMatched ||
      !activeCitation.quote ||
      !textLayerRef.current ||
      textLayerRenderKey !== renderKey ||
      renderError
    ) {
      setHighlightRects([]);
      setIsExactMatch(false);
      return;
    }

    const range = findRangeForQuote(textLayerRef.current, activeCitation.quote);

    if (!range) {
      setHighlightRects([]);
      setIsExactMatch(false);
      return;
    }

    const containerRect = textLayerRef.current.getBoundingClientRect();
    const clientRects = range.getClientRects();
    const rects: HighlightRect[] = [];

    if (clientRects && clientRects.length > 0) {
      for (let i = 0; i < clientRects.length; i++) {
        const r = clientRects[i];
        if (r.width > 0 && r.height > 0) {
          const left = r.left - containerRect.left;
          const top = r.top - containerRect.top;
          const hasContainerBounds =
            containerRect.width > 0 && containerRect.height > 0;
          const isWithinBounds =
            !hasContainerBounds ||
            (left >= -2 &&
              top >= -2 &&
              left + r.width <= containerRect.width + 5 &&
              top + r.height <= containerRect.height + 5);

          if (isWithinBounds) {
            rects.push({
              left: Math.max(0, left),
              top: Math.max(0, top),
              width: r.width,
              height: r.height,
            });
          }
        }
      }
    }

    if (rects.length > 0) {
      setHighlightRects(rects);
      setIsExactMatch(true);
    } else {
      setHighlightRects([]);
      setIsExactMatch(false);
    }
  }, [
    activeCitation,
    paper,
    currentPage,
    renderError,
    servedDocument,
    textLayerRenderKey,
    renderKey,
  ]);

  useEffect(() => {
    const frameId = requestAnimationFrame(() => {
      recomputeHighlights();
    });
    return () => {
      cancelAnimationFrame(frameId);
    };
  }, [recomputeHighlights, textLayerReady, scale, rotation]);

  // Recompute on window/container resize
  useEffect(() => {
    window.addEventListener("resize", recomputeHighlights);
    return () => {
      window.removeEventListener("resize", recomputeHighlights);
    };
  }, [recomputeHighlights]);

  const isCurrentPageCited = Boolean(
    activeCitation &&
    activeCitation.paper_id === paper?.id &&
    activeCitation.page_number === currentPage,
  );

  const highlightStatus: "exact" | "unavailable" =
    isCurrentPageCited &&
    textLayerRenderKey === renderKey &&
    isExactMatch &&
    highlightRects.length > 0
      ? "exact"
      : "unavailable";

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
                  : "Verbatim match"}
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
            data-testid="pdf-text-layer"
            onMouseUp={captureTextSelection}
            onKeyUp={captureTextSelection}
            className="textLayer absolute inset-0 select-text overflow-hidden opacity-0 pointer-events-auto"
          />

          {/* Exact Range.getClientRects() highlights for cited page */}
          {isCurrentPageCited &&
            highlightStatus === "exact" &&
            highlightRects.map((rect, idx) => (
              <div
                key={`exact-highlight-${idx}`}
                data-testid="evidence-highlight"
                style={{
                  position: "absolute",
                  left: `${rect.left}px`,
                  top: `${rect.top}px`,
                  width: `${rect.width}px`,
                  height: `${rect.height}px`,
                  backgroundColor: "rgba(245, 158, 11, 0.35)",
                  border: "2px solid #d97706",
                  boxShadow: "0 0 6px rgba(245, 158, 11, 0.4)",
                  borderRadius: "2px",
                  pointerEvents: "none",
                  zIndex: 20,
                }}
              />
            ))}
        </div>
      </div>
    </div>
  );
}
