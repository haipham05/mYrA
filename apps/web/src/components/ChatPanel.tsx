"use client";

import { useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { visit } from "unist-util-visit";
import type { PhrasingContent, Root } from "mdast";
import ConversationNavigator from "@/components/ConversationNavigator";
import DiscoveryImportControls from "@/components/DiscoveryImportControls";
import type {
  AssistantApprovalResponse,
  AssistantIntent,
  AssistantResultType,
  AssistantRunResponse,
  Citation,
  Conversation,
  Message,
  SourceSelection,
  VisualSelection,
  VisualSourceReference,
} from "@/types";

interface ComparisonExcerptView {
  quote: string;
  paperTitle?: string;
  citation: Citation;
}

interface ComparisonCellView {
  paperId: string;
  dimension: string;
  status: string;
  message?: string;
  excerpts: ComparisonExcerptView[];
}

interface ComparisonMatrixView {
  paperIds: string[];
  dimensions: string[];
  cells: ComparisonCellView[];
}

interface BenchmarkComparisonView {
  leftPaperId: string;
  rightPaperId: string;
  leftResult: string;
  rightResult: string;
  status: string;
  reasons: string[];
  evidenceIds: string[];
}

interface PaperFindingView {
  paperId: string;
  title: string;
  summary: string;
  evidenceAvailable: boolean;
}

interface DiscoveryCandidateView {
  catalog: "openalex" | "arxiv";
  title: string;
  authors: string[];
  publicationYear?: number;
  doi?: string;
  arxivId?: string;
  abstract?: string;
  sourceUrl: string;
  pdfUrl?: string;
  openAccess?: boolean;
  possibleDuplicate: boolean;
  candidatePayload: Record<string, unknown>;
}

interface NoteCardView {
  id?: string;
  title: string;
  content: string;
  type?: string;
  status?: string;
  version?: number;
  sources?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function failureMessage(result: Message["assistantResult"]): {
  heading: string;
  detail: string;
} | null {
  if (!result) return null;
  const warnings = result.warnings.map((warning) => warning.toUpperCase());
  const payload = isRecord(result.structured_payload)
    ? result.structured_payload
    : {};
  const errorCode =
    typeof payload.safe_error === "string"
      ? payload.safe_error.toUpperCase()
      : warnings.find((warning) =>
          [
            "BUDGET_DENIED",
            "PROVIDER_UNAVAILABLE",
            "PROVIDER_RATE_LIMITED",
            "SOURCE_CHANGED",
            "APPROVAL_SOURCE_CHANGED",
            "APPROVAL_NO_LONGER_VALID",
          ].includes(warning),
        );

  if (errorCode === "BUDGET_DENIED") {
    return {
      heading: "Spending limit reached",
      detail:
        "This step stopped before its paid model call. Reduce the request scope or check your remaining allowance before retrying.",
    };
  }
  if (
    errorCode === "PROVIDER_UNAVAILABLE" ||
    errorCode === "PROVIDER_RATE_LIMITED"
  ) {
    return {
      heading: "Research model unavailable",
      detail:
        "The model service could not complete this request. No answer was produced; you can try again later.",
    };
  }
  if (
    errorCode === "APPROVAL_SOURCE_CHANGED" ||
    errorCode === "APPROVAL_NO_LONGER_VALID" ||
    result.result_type === "approval_invalidated" ||
    result.result_type === "graph_index_approval_invalid"
  ) {
    return {
      heading: "The source changed",
      detail:
        "The saved approval no longer matches the current paper or request. Review the current source and submit the action again.",
    };
  }
  if (
    errorCode === "SOURCE_CHANGED" ||
    result.result_type === "source_selection_unavailable"
  ) {
    return {
      heading: "Selected passage unavailable",
      detail:
        "The selected source no longer matches the current paper. Review the latest paper, then select the passage again or ask a new question.",
    };
  }
  if (result.result_type === "routing_unavailable") {
    return {
      heading: "Could not route this request",
      detail:
        "The research model could not choose an action. No research action was started; try again later or choose a more specific request.",
    };
  }
  if (result.result_type === "action_outcome_unknown") {
    return {
      heading: "The result could not be confirmed",
      detail:
        "mYrA did not repeat the action automatically. Check the relevant paper or saved output before trying again.",
    };
  }
  if (result.result_type === "clarification") {
    const missing = isRecord(result.structured_payload)
      ? result.structured_payload.missing_information
      : null;
    if (Array.isArray(missing) && missing.includes("ready_paper")) {
      return {
        heading: "Choose a ready paper",
        detail:
          "Select one paper that has finished processing, then submit the request again.",
      };
    }
    if (
      Array.isArray(missing) &&
      missing.includes("select_or_identify_paper")
    ) {
      return {
        heading: "Choose which paper to use",
        detail:
          "Select a paper from this project or clarify its title before continuing.",
      };
    }
    if (
      Array.isArray(missing) &&
      missing.some((item) =>
        ["paper", "paper_scope", "selected_paper_ids", "scope"].includes(
          String(item),
        ),
      )
    ) {
      return {
        heading: "Choose the research scope",
        detail:
          "Choose the current paper, selected papers, or the whole project before continuing.",
      };
    }
  }
  return null;
}

function runStatusLabel(status: AssistantRunResponse["status"]): string {
  const labels: Record<AssistantRunResponse["status"], string> = {
    QUEUED: "Waiting for a worker",
    RUNNING: "Research in progress",
    NEEDS_INPUT: "More information needed",
    AWAITING_APPROVAL: "Waiting for your approval",
    SUCCEEDED: "Completed",
    FAILED: "Not completed",
    CANCELLED: "Cancelled",
  };
  return labels[status];
}

function isCitation(value: unknown): value is Citation {
  return (
    isRecord(value) &&
    typeof value.citation_index === "number" &&
    typeof value.evidence_id === "string" &&
    typeof value.paper_id === "string" &&
    typeof value.page_number === "number" &&
    typeof value.quote === "string" &&
    Array.isArray(value.bounding_boxes)
  );
}

function readComparisonMatrix(payload: unknown): ComparisonMatrixView | null {
  if (!isRecord(payload)) return null;
  const matrix = payload.matrix;
  if (
    !isRecord(matrix) ||
    !Array.isArray(matrix.paper_ids) ||
    !matrix.paper_ids.every((id) => typeof id === "string") ||
    !Array.isArray(matrix.dimensions) ||
    !matrix.dimensions.every((dimension) => typeof dimension === "string") ||
    !Array.isArray(matrix.cells)
  ) {
    return null;
  }

  const cells = matrix.cells.flatMap((value): ComparisonCellView[] => {
    if (
      !isRecord(value) ||
      typeof value.paper_id !== "string" ||
      typeof value.dimension !== "string" ||
      typeof value.status !== "string"
    ) {
      return [];
    }
    const excerpts = Array.isArray(value.excerpts)
      ? value.excerpts.flatMap((excerpt): ComparisonExcerptView[] => {
          if (!isRecord(excerpt) || !isRecord(excerpt.evidence)) return [];
          const citation = excerpt.citation;
          if (!isCitation(citation)) return [];
          const evidence = excerpt.evidence;
          const quote =
            typeof evidence.quote === "string"
              ? evidence.quote
              : citation.quote;
          return [
            {
              quote,
              paperTitle:
                typeof evidence.paper_title === "string"
                  ? evidence.paper_title
                  : undefined,
              citation,
            },
          ];
        })
      : [];
    return [
      {
        paperId: value.paper_id,
        dimension: value.dimension,
        status: value.status,
        message: typeof value.message === "string" ? value.message : undefined,
        excerpts,
      },
    ];
  });

  return {
    paperIds: matrix.paper_ids,
    dimensions: matrix.dimensions,
    cells,
  };
}

function readBenchmarkComparisons(payload: unknown): BenchmarkComparisonView[] {
  if (!isRecord(payload) || !isRecord(payload.synthesis)) return [];
  const comparisons = payload.synthesis.benchmark_comparisons;
  if (!Array.isArray(comparisons)) return [];
  return comparisons.flatMap((value): BenchmarkComparisonView[] => {
    if (
      !isRecord(value) ||
      typeof value.left_paper_id !== "string" ||
      typeof value.right_paper_id !== "string" ||
      typeof value.left_result !== "string" ||
      typeof value.right_result !== "string" ||
      !isRecord(value.comparability) ||
      typeof value.comparability.status !== "string" ||
      !Array.isArray(value.evidence_ids) ||
      !value.evidence_ids.every((id) => typeof id === "string")
    ) {
      return [];
    }
    return [
      {
        leftPaperId: value.left_paper_id,
        rightPaperId: value.right_paper_id,
        leftResult: value.left_result,
        rightResult: value.right_result,
        status: value.comparability.status,
        reasons: Array.isArray(value.comparability.reasons)
          ? value.comparability.reasons.filter(
              (reason): reason is string => typeof reason === "string",
            )
          : [],
        evidenceIds: value.evidence_ids,
      },
    ];
  });
}

function readDiscoveryCandidates(payload: unknown): DiscoveryCandidateView[] {
  if (!isRecord(payload) || !Array.isArray(payload.items)) return [];
  return payload.items.flatMap((item): DiscoveryCandidateView[] => {
    if (
      !isRecord(item) ||
      (item.catalog !== "openalex" && item.catalog !== "arxiv") ||
      typeof item.title !== "string" ||
      typeof item.source_url !== "string"
    ) {
      return [];
    }
    let sourceUrl: URL;
    try {
      sourceUrl = new URL(item.source_url);
    } catch {
      return [];
    }
    const allowedHosts =
      item.catalog === "openalex"
        ? new Set(["openalex.org", "www.openalex.org"])
        : new Set(["arxiv.org", "www.arxiv.org"]);
    if (
      sourceUrl.protocol !== "https:" ||
      !allowedHosts.has(sourceUrl.hostname)
    ) {
      return [];
    }
    let pdfUrl: string | undefined;
    if (typeof item.pdf_url === "string") {
      try {
        const parsedPdfUrl = new URL(item.pdf_url);
        if (parsedPdfUrl.protocol === "https:" && parsedPdfUrl.hostname) {
          pdfUrl = parsedPdfUrl.toString();
        }
      } catch {
        pdfUrl = undefined;
      }
    }
    return [
      {
        catalog: item.catalog,
        title: item.title,
        authors: Array.isArray(item.authors)
          ? item.authors.filter(
              (author): author is string => typeof author === "string",
            )
          : [],
        publicationYear:
          typeof item.publication_year === "number"
            ? item.publication_year
            : undefined,
        doi: typeof item.doi === "string" ? item.doi : undefined,
        arxivId: typeof item.arxiv_id === "string" ? item.arxiv_id : undefined,
        abstract: typeof item.abstract === "string" ? item.abstract : undefined,
        sourceUrl: sourceUrl.toString(),
        pdfUrl,
        openAccess:
          typeof item.open_access === "boolean" ? item.open_access : undefined,
        possibleDuplicate: item.possible_duplicate === true,
        candidatePayload: item,
      },
    ];
  });
}

interface ChatPanelProps {
  messages: Message[];
  isLoading: boolean;
  routedIntent?: AssistantIntent | null;
  error?: string | null;
  onDismissError?: () => void;
  onSendMessage: (content: string) => Promise<void>;
  onCitationClick: (citation: Citation) => void;
  activeCitation: Citation | null;
  disabled: boolean;
  conversations?: Conversation[];
  activeConversation?: Conversation | null;
  onSelectConversation?: (conv: Conversation) => void;
  onCreateConversation?: (title?: string) => Promise<void>;
  onRenameConversation?: (id: string, newTitle: string) => Promise<void>;
  onArchiveConversation?: (id: string, isArchived: boolean) => Promise<void>;
  onDeleteConversation?: (id: string) => Promise<void>;
  activeRun?: AssistantRunResponse | null;
  pendingAction?: AssistantApprovalResponse | null;
  onCancelRun?: (runId: string) => Promise<void>;
  onResumeRun?: (runId: string, input: string) => Promise<void>;
  onDecideAction?: (actionId: string, approve: boolean) => Promise<void>;
  onProposeDiscoveryImport?: (
    runId: string,
    candidate: Record<string, unknown>,
  ) => Promise<AssistantApprovalResponse>;
  onDecideDiscoveryImport?: (
    actionId: string,
    approve: boolean,
  ) => Promise<AssistantApprovalResponse>;
  sourceSelection?: SourceSelection | null;
  onClearSourceSelection?: () => void;
  visualSelection?: VisualSelection | null;
  onClearVisualSelection?: () => void;
  onVisualSourceClick?: (source: VisualSourceReference) => void;
}

export default function ChatPanel({
  messages,
  isLoading,
  routedIntent,
  error,
  onDismissError,
  onSendMessage,
  onCitationClick,
  activeCitation,
  disabled,
  conversations,
  activeConversation,
  onSelectConversation,
  onCreateConversation,
  onRenameConversation,
  onArchiveConversation,
  onDeleteConversation,
  activeRun,
  pendingAction,
  onCancelRun,
  onResumeRun,
  onDecideAction,
  onProposeDiscoveryImport,
  onDecideDiscoveryImport,
  sourceSelection,
  onClearSourceSelection,
  visualSelection,
  onClearVisualSelection,
  onVisualSourceClick,
}: ChatPanelProps) {
  const [input, setInput] = useState("");
  const [resumeInput, setResumeInput] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || disabled) return;
    const content = input.trim();
    setInput("");
    await onSendMessage(content);
  };

  const renderContent = (content: string, citations: Citation[]) => (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkMath, remarkCitationLinks]}
      rehypePlugins={[[rehypeKatex, { throwOnError: false }]]}
      skipHtml
      urlTransform={(url) =>
        url.startsWith("myra-citation:") ? url : defaultUrlTransform(url)
      }
      components={{
        h1: ({ children }) => (
          <h1 className="my-3 text-lg font-semibold">{children}</h1>
        ),
        h2: ({ children }) => (
          <h2 className="my-3 text-base font-semibold">{children}</h2>
        ),
        h3: ({ children }) => (
          <h3 className="my-2 font-semibold">{children}</h3>
        ),
        p: ({ children }) => (
          <p className="my-2 whitespace-pre-wrap">{children}</p>
        ),
        ul: ({ children }) => (
          <ul className="my-2 list-disc space-y-1 pl-5">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="my-2 list-decimal space-y-1 pl-5">{children}</ol>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-2 border-l-2 border-zinc-300 pl-3 text-zinc-600">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="my-3 overflow-x-auto">
            <table className="min-w-full border-collapse text-left">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-zinc-300 bg-zinc-50 px-2 py-1">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-zinc-300 px-2 py-1 align-top">
            {children}
          </td>
        ),
        code: ({ children }) => (
          <code className="rounded bg-zinc-200 px-1 py-0.5 font-mono text-[0.9em]">
            {children}
          </code>
        ),
        a: ({ href, children }) => {
          const citationMatch = href?.match(/^myra-citation:(\d+)$/);
          if (!citationMatch) {
            return (
              <a
                href={href}
                target="_blank"
                rel="noreferrer"
                className="text-blue-700 underline"
              >
                {children}
              </a>
            );
          }
          const citationIndex = Number(citationMatch[1]);
          const citation = citations.find(
            (item) => item.citation_index === citationIndex,
          );
          if (!citation) return <>{children}</>;
          const isSelected = activeCitation?.citation_index === citationIndex;
          return (
            <button
              type="button"
              onClick={() => onCitationClick(citation)}
              className={`mx-0.5 inline-flex cursor-pointer items-center rounded px-1.5 py-0.5 text-xs font-semibold ${
                isSelected
                  ? "bg-amber-400 text-amber-950 ring-2 ring-amber-500"
                  : "bg-blue-100 text-blue-800 hover:bg-blue-200"
              }`}
              title={`Click to view Page ${citation.page_number} evidence`}
            >
              {children}
            </button>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );

  const renderComparison = (message: Message) => {
    const result = message.assistantResult;
    if (result?.result_type !== "comparison") return null;
    if (
      isRecord(result.structured_payload) &&
      Array.isArray(result.structured_payload.paper_findings)
    ) {
      const findings = result.structured_payload.paper_findings.flatMap(
        (value): PaperFindingView[] =>
          isRecord(value) &&
          typeof value.title === "string" &&
          typeof value.summary === "string" &&
          typeof value.evidence_available === "boolean"
            ? [
                {
                  paperId:
                    typeof value.paper_id === "string"
                      ? value.paper_id
                      : value.title,
                  title: value.title,
                  summary: value.summary,
                  evidenceAvailable: value.evidence_available,
                },
              ]
            : [],
      );
      if (findings.length > 0) {
        return (
          <div
            className="space-y-3"
            aria-label="Paper comparison"
            role="region"
          >
            {findings.map((finding) => (
              <section
                key={finding.paperId}
                className="rounded border border-zinc-200 p-3"
              >
                <h3 className="text-sm font-semibold">{finding.title}</h3>
                {!finding.evidenceAvailable && (
                  <p className="text-xs text-zinc-500">
                    No relevant evidence found.
                  </p>
                )}
                <div className="mt-2 text-sm">
                  {renderContent(finding.summary, result.citations)}
                </div>
              </section>
            ))}
          </div>
        );
      }
    }
    const matrix = readComparisonMatrix(result.structured_payload);
    if (!matrix) return null;
    const benchmarkComparisons = readBenchmarkComparisons(
      result.structured_payload,
    );
    const paperTitles = new Map(
      matrix.cells.map((cell) => [
        cell.paperId,
        cell.excerpts.find((excerpt) => excerpt.paperTitle)?.paperTitle ??
          cell.paperId,
      ]),
    );

    return (
      <div className="space-y-3">
        <p className="text-xs font-medium text-amber-800" role="note">
          Candidate source excerpts only; these are not extracted or
          fact-checked claims.
        </p>
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-xs">
            <thead>
              <tr>
                <th className="border border-zinc-300 bg-zinc-50 px-2 py-1">
                  Paper
                </th>
                {matrix.dimensions.map((dimension) => (
                  <th
                    key={dimension}
                    className="border border-zinc-300 bg-zinc-50 px-2 py-1"
                  >
                    {dimension.replaceAll("_", " ")}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {matrix.paperIds.map((paperId) => {
                const paperCells = matrix.cells.filter(
                  (cell) => cell.paperId === paperId,
                );
                const title = paperCells
                  .flatMap((cell) => cell.excerpts)
                  .find((excerpt) => excerpt.paperTitle)?.paperTitle;
                return (
                  <tr key={paperId}>
                    <th className="border border-zinc-300 px-2 py-1 align-top font-medium">
                      {title ?? paperId}
                    </th>
                    {matrix.dimensions.map((dimension) => {
                      const cell = paperCells.find(
                        (candidate) => candidate.dimension === dimension,
                      );
                      return (
                        <td
                          key={dimension}
                          className="border border-zinc-300 px-2 py-1 align-top"
                        >
                          {!cell || cell.excerpts.length === 0 ? (
                            <span className="text-zinc-500">
                              {cell?.message ??
                                "Not reported in retrieved evidence."}
                            </span>
                          ) : (
                            <ul className="space-y-2">
                              {cell.excerpts.map((excerpt) => (
                                <li key={excerpt.citation.evidence_id}>
                                  <blockquote className="border-l-2 border-zinc-300 pl-2">
                                    {excerpt.quote}
                                  </blockquote>
                                  <button
                                    type="button"
                                    onClick={() =>
                                      onCitationClick(excerpt.citation)
                                    }
                                    className="mt-1 rounded bg-blue-100 px-1.5 py-0.5 font-semibold text-blue-800 hover:bg-blue-200"
                                    title={`Click to view Page ${excerpt.citation.page_number} evidence`}
                                  >
                                    Page {excerpt.citation.page_number} ·[
                                    {excerpt.citation.citation_index}]
                                  </button>
                                </li>
                              ))}
                            </ul>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {benchmarkComparisons.length > 0 && (
          <section aria-label="Benchmark comparisons" className="space-y-2">
            <h3 className="text-sm font-semibold">
              Reported benchmark results
            </h3>
            {benchmarkComparisons.map((comparison, index) => (
              <div
                key={`${comparison.evidenceIds.join("-")}-${index}`}
                className="rounded border border-zinc-200 p-2 text-xs"
              >
                <p className="font-medium">
                  {paperTitles.get(comparison.leftPaperId) ??
                    comparison.leftPaperId}
                  : {comparison.leftResult} ·{" "}
                  {paperTitles.get(comparison.rightPaperId) ??
                    comparison.rightPaperId}
                  : {comparison.rightResult}
                </p>
                <p className="text-zinc-600">
                  {comparison.status === "directly_comparable"
                    ? "Reported under matching benchmark conditions; values shown without ranking."
                    : "Not directly comparable."}
                </p>
                {comparison.reasons.map((reason) => (
                  <p key={reason} className="text-zinc-600">
                    {reason}
                  </p>
                ))}
                <div className="mt-1 flex gap-2">
                  {comparison.evidenceIds.map((evidenceId) => {
                    const citation = result.citations.find(
                      (item) => item.evidence_id === evidenceId,
                    );
                    return citation ? (
                      <button
                        key={evidenceId}
                        type="button"
                        onClick={() => onCitationClick(citation)}
                        className="rounded bg-blue-100 px-1.5 py-0.5 font-semibold text-blue-800 hover:bg-blue-200"
                      >
                        Page {citation.page_number} ·[{citation.citation_index}]
                      </button>
                    ) : null;
                  })}
                </div>
              </div>
            ))}
          </section>
        )}
        {result.warnings
          .filter(
            (warning) =>
              typeof warning === "string" &&
              !warning.toLowerCase().includes("candidate source excerpts"),
          )
          .map((warning) => (
            <p key={warning} className="text-xs text-amber-800">
              {warning}
            </p>
          ))}
        {renderContent(message.content, message.citations)}
      </div>
    );
  };

  const renderVisualAnalysis = (message: Message) => {
    const result = message.assistantResult;
    if (result?.result_type !== "visual_analysis") return null;
    const payload = result.structured_payload;
    if (!isRecord(payload) || !isRecord(payload.analysis)) return null;
    const analysis = payload.analysis;
    const source = payload.visual_source;
    if (
      !isRecord(source) ||
      source.source_kind !== "visual" ||
      source.text_citation !== false ||
      typeof source.paper_id !== "string" ||
      typeof source.page_number !== "number" ||
      typeof source.document_sha256 !== "string" ||
      !isRecord(source.crop_box_normalized_top_left)
    ) {
      return null;
    }
    const observations = Array.isArray(analysis.observations)
      ? analysis.observations.flatMap((item) =>
          isRecord(item) && typeof item.statement === "string"
            ? [item.statement]
            : [],
        )
      : [];
    const readings = Array.isArray(analysis.readings)
      ? analysis.readings.flatMap((item) =>
          isRecord(item) &&
          typeof item.label === "string" &&
          typeof item.value === "string" &&
          (item.kind === "direct_reading" || item.kind === "plot_estimate")
            ? [
                {
                  label: item.label,
                  value: item.value,
                  kind: item.kind,
                  unit: typeof item.unit === "string" ? item.unit : null,
                },
              ]
            : [],
        )
      : [];
    const uncertainties = Array.isArray(analysis.uncertainty_notes)
      ? analysis.uncertainty_notes.filter(
          (item): item is string => typeof item === "string",
        )
      : [];

    return (
      <section className="space-y-2" aria-label="Visual analysis">
        <p className="text-xs font-medium text-blue-900">
          Figure analysis — visual interpretation, not a text quotation.
        </p>
        {observations.length > 0 && (
          <ul className="list-disc space-y-1 pl-5">
            {observations.map((item, index) => (
              <li key={`visual-observation-${index}`}>{item}</li>
            ))}
          </ul>
        )}
        {readings.length > 0 && (
          <ul className="space-y-1">
            {readings.map((item, index) => (
              <li key={`visual-reading-${index}`}>
                <span className="font-medium">{item.label}:</span> {item.value}
                {item.unit ? ` ${item.unit}` : ""}
                <span className="ml-1 text-xs text-zinc-600">
                  (
                  {item.kind === "plot_estimate"
                    ? "estimated from plot"
                    : "read directly"}
                  )
                </span>
              </li>
            ))}
          </ul>
        )}
        {typeof analysis.interpretation === "string" &&
          analysis.interpretation && <p>{analysis.interpretation}</p>}
        {uncertainties.length > 0 && (
          <div className="rounded bg-amber-50 p-2 text-xs text-amber-900">
            <p className="font-medium">Uncertainty</p>
            <ul className="list-disc pl-5">
              {uncertainties.map((item, index) => (
                <li key={`visual-uncertainty-${index}`}>{item}</li>
              ))}
            </ul>
          </div>
        )}
        {onVisualSourceClick && (
          <button
            type="button"
            className="text-xs font-medium text-blue-800 underline"
            onClick={() =>
              onVisualSourceClick(source as unknown as VisualSourceReference)
            }
          >
            Open original page {source.page_number}
          </button>
        )}
      </section>
    );
  };

  const renderDiscovery = (message: Message) => {
    const result = message.assistantResult;
    if (result?.result_type !== "discovery_results") return null;
    const payload = result.structured_payload;
    const candidates = readDiscoveryCandidates(payload);
    const runId =
      isRecord(payload) && typeof payload.run_id === "string"
        ? payload.run_id
        : null;
    const sourceErrors =
      isRecord(payload) && isRecord(payload.source_errors)
        ? Object.keys(payload.source_errors)
        : [];
    return (
      <div className="space-y-3">
        <p className="text-xs font-medium text-amber-800" role="note">
          Metadata only. Abstracts are catalog descriptions, not indexed
          full-text evidence. Nothing was downloaded or added to your library.
        </p>
        {sourceErrors.map((source) => (
          <p key={source} className="text-xs text-amber-800">
            {source} search was unavailable.
          </p>
        ))}
        {candidates.map((candidate) => (
          <article
            key={`${candidate.catalog}:${candidate.sourceUrl}`}
            className="rounded border border-zinc-200 p-3"
          >
            <div className="flex flex-wrap items-start justify-between gap-2">
              <h3 className="font-medium">{candidate.title}</h3>
              <span className="rounded bg-zinc-100 px-2 py-0.5 text-[11px]">
                {candidate.catalog}
                {candidate.openAccess === true
                  ? " · open access"
                  : candidate.openAccess === false
                    ? " · not open access"
                    : " · access unknown"}
              </span>
            </div>
            {(candidate.authors.length > 0 || candidate.publicationYear) && (
              <p className="mt-1 text-xs text-zinc-600">
                {[candidate.authors.join(", "), candidate.publicationYear]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
            {(candidate.doi || candidate.arxivId) && (
              <p className="mt-1 text-xs text-zinc-600">
                {candidate.doi ? `DOI: ${candidate.doi}` : ""}
                {candidate.doi && candidate.arxivId ? " · " : ""}
                {candidate.arxivId ? `arXiv: ${candidate.arxivId}` : ""}
              </p>
            )}
            {candidate.possibleDuplicate && (
              <p className="mt-1 text-xs text-amber-800">
                Possible title match in these results; inspect before importing.
              </p>
            )}
            {candidate.abstract && (
              <p className="mt-2 whitespace-pre-wrap text-xs text-zinc-700">
                {candidate.abstract}
              </p>
            )}
            <a
              className="mt-2 inline-block text-xs text-blue-700 underline"
              href={candidate.sourceUrl}
              target="_blank"
              rel="noreferrer"
            >
              Open catalog record
            </a>
            {onProposeDiscoveryImport && onDecideDiscoveryImport && (
              <DiscoveryImportControls
                runId={runId}
                candidate={candidate.candidatePayload}
                pdfUrl={candidate.pdfUrl}
                openAccess={candidate.openAccess}
                disabled={isLoading}
                onPropose={onProposeDiscoveryImport}
                onDecide={onDecideDiscoveryImport}
              />
            )}
          </article>
        ))}
        {renderContent(message.content, message.citations)}
      </div>
    );
  };

  const renderResearchDiscovery = (message: Message) => {
    const result = message.assistantResult;
    if (
      result?.result_type !== "research_draft" ||
      !result.available_actions.includes("discover") ||
      !isRecord(result.structured_payload) ||
      typeof result.structured_payload.discovery_query !== "string"
    ) {
      return null;
    }

    const query = result.structured_payload.discovery_query;
    return (
      <div className="mt-3 rounded border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
        <p>
          This evidence gap can be explored through paper discovery. Search
          shows catalog metadata only; importing a result requires your
          approval. After its paper finishes indexing, ask mYrA to continue the
          research.
        </p>
        <button
          type="button"
          disabled={disabled || isLoading}
          onClick={() =>
            void onSendMessage(`Find academic papers about: ${query}`)
          }
          className="mt-2 rounded bg-amber-900 px-3 py-1.5 font-medium text-white disabled:opacity-50"
        >
          Discover papers for this gap
        </button>
        <div className="mt-3 text-zinc-900">
          {renderContent(message.content, message.citations)}
        </div>
      </div>
    );
  };

  const renderReadingBrief = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (result?.result_type !== "reading_brief") return null;
    const payload = result.structured_payload;
    if (!isRecord(payload) || !Array.isArray(payload.sections)) return null;
    const sections = payload.sections.flatMap((value) =>
      isRecord(value) &&
      typeof value.title === "string" &&
      typeof value.content === "string"
        ? [
            {
              title: value.title,
              content: value.content,
              citationIndexes: Array.isArray(value.citation_indexes)
                ? value.citation_indexes.filter(
                    (index): index is number => typeof index === "number",
                  )
                : [],
            },
          ]
        : [],
    );
    if (sections.length === 0) return null;
    return (
      <section className="space-y-3" aria-label="Paper reading brief">
        {sections.map((section) => (
          <div key={section.title}>
            <h3 className="font-semibold">{section.title}</h3>
            <div>{renderContent(section.content, message.citations)}</div>
            <div className="mt-1 flex flex-wrap gap-1">
              {section.citationIndexes.map((index) => {
                const citation = message.citations.find(
                  (item) => item.citation_index === index,
                );
                return citation ? (
                  <button
                    key={`${section.title}:${index}`}
                    type="button"
                    onClick={() => onCitationClick(citation)}
                    className="rounded bg-blue-100 px-1.5 py-0.5 text-xs font-semibold text-blue-800 hover:bg-blue-200"
                  >
                    Source [{index}] · page {citation.page_number}
                  </button>
                ) : null;
              })}
            </div>
          </div>
        ))}
      </section>
    );
  };

  const renderClaimVerification = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (result?.result_type !== "claim_verification") return null;
    const payload = result.structured_payload;
    if (
      !isRecord(payload) ||
      typeof payload.verdict !== "string" ||
      typeof payload.explanation !== "string"
    ) {
      return null;
    }
    return (
      <section className="space-y-2" aria-label="Claim verification">
        <p className="font-semibold capitalize">
          Claim assessment: {payload.verdict.replaceAll("_", " ")}
        </p>
        {typeof payload.claim === "string" && (
          <blockquote className="border-l-2 border-zinc-300 pl-3 text-zinc-700">
            {payload.claim}
          </blockquote>
        )}
        <p>{payload.explanation}</p>
        {typeof payload.scope_note === "string" && (
          <p className="text-xs text-amber-800">{payload.scope_note}</p>
        )}
        <div className="flex flex-wrap gap-1">
          {message.citations.map((citation) => (
            <button
              key={citation.evidence_id}
              type="button"
              onClick={() => onCitationClick(citation)}
              className="rounded bg-blue-100 px-1.5 py-0.5 text-xs font-semibold text-blue-800 hover:bg-blue-200"
            >
              Source [{citation.citation_index}] · page {citation.page_number}
            </button>
          ))}
        </div>
      </section>
    );
  };

  const renderNotes = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (
      !result ||
      (result.result_type !== "notes_list" && result.result_type !== "note")
    ) {
      return null;
    }
    const payload = result.structured_payload;
    if (!isRecord(payload)) return null;
    const items =
      result.result_type === "notes_list"
        ? Array.isArray(payload.items)
          ? payload.items
          : []
        : payload.item
          ? [payload.item]
          : [];
    const notes = items.flatMap((value): NoteCardView[] =>
      isRecord(value) &&
      typeof value.title === "string" &&
      typeof value.content === "string"
        ? [
            {
              id: typeof value.id === "string" ? value.id : undefined,
              title: value.title,
              content: value.content,
              type: typeof value.type === "string" ? value.type : undefined,
              status:
                typeof value.status === "string" ? value.status : undefined,
              version:
                typeof value.version === "number" ? value.version : undefined,
              sources: value.sources,
            },
          ]
        : [],
    );
    if (items.length > 0 && notes.length === 0) return null;
    return (
      <section className="space-y-2" aria-label="Research notes">
        {result.result_type === "notes_list" && (
          <p className="text-xs text-zinc-600">
            {typeof payload.total === "number" ? payload.total : notes.length}{" "}
            active note(s)
          </p>
        )}
        {notes.map((note, index) => (
          <article
            key={
              typeof note.id === "string" ? note.id : `${note.title}-${index}`
            }
            className="rounded border border-zinc-200 bg-white p-3"
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="font-semibold">{note.title}</h3>
              <span className="text-xs text-zinc-500">
                {[
                  note.type,
                  note.status,
                  typeof note.version === "number" ? `v${note.version}` : null,
                ]
                  .filter((part): part is string => typeof part === "string")
                  .join(" · ")}
              </span>
            </div>
            <p className="mt-1 whitespace-pre-wrap">{note.content}</p>
            {Array.isArray(note.sources) && note.sources.length > 0 && (
              <p className="mt-2 text-xs text-zinc-600">
                Source context:{" "}
                {note.sources
                  .flatMap((source) =>
                    isRecord(source) && typeof source.page_number === "number"
                      ? [`page ${source.page_number}`]
                      : [],
                  )
                  .join(", ") || `${note.sources.length} reference(s)`}
              </p>
            )}
          </article>
        ))}
        {notes.length === 0 && <p>No active research notes.</p>}
      </section>
    );
  };

  const renderReport = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (result?.result_type !== "research_report") return null;
    const payload = result.structured_payload;
    if (!isRecord(payload) || typeof payload.report_markdown !== "string") {
      return null;
    }
    const sources = Array.isArray(payload.source_manifest)
      ? payload.source_manifest.flatMap((source) =>
          isRecord(source) &&
          typeof source.citation_index === "number" &&
          typeof source.page_number === "number"
            ? [source]
            : [],
        )
      : [];
    return (
      <section className="space-y-3" aria-label="Research report">
        <div className="rounded border border-zinc-200 bg-white p-3">
          {renderContent(payload.report_markdown, message.citations)}
        </div>
        {sources.length > 0 && (
          <div className="flex flex-wrap gap-1" aria-label="Report sources">
            {sources.map((source, index) => {
              const citation = message.citations.find(
                (item) => item.citation_index === source.citation_index,
              );
              return citation ? (
                <button
                  key={`${source.citation_index}-${index}`}
                  type="button"
                  onClick={() => onCitationClick(citation)}
                  className="rounded bg-blue-100 px-1.5 py-0.5 text-xs font-semibold text-blue-800 hover:bg-blue-200"
                >
                  {typeof source.paper_title === "string"
                    ? source.paper_title
                    : "Source"}{" "}
                  · page {citation.page_number}
                </button>
              ) : null;
            })}
          </div>
        )}
        <p className="text-xs text-zinc-500">
          {payload.saved === true ? "Saved report" : "Draft report — not saved"}
          {typeof payload.scope === "string" ? ` · ${payload.scope} scope` : ""}
        </p>
        {result.artifact_ids.length > 0 && (
          <p className="text-xs text-zinc-600" aria-label="Report artifacts">
            Saved artifact reference(s): {result.artifact_ids.join(", ")}
          </p>
        )}
      </section>
    );
  };

  const renderGraphIndex = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (result?.result_type !== "graph_index_jobs") return null;
    const payload = result.structured_payload;
    if (!isRecord(payload)) return null;
    const counts = [
      ["Queued", payload.queued_count],
      ["Already queued or skipped", payload.skipped_count],
      ["Completed", payload.completed_count],
      ["Failed", payload.failed_count],
    ].filter(
      (entry): entry is [string, number] => typeof entry[1] === "number",
    );
    if (counts.length === 0) return null;
    return (
      <section className="space-y-2" aria-label="Graph indexing jobs">
        <p className="font-medium">Optional graph indexing</p>
        <dl className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {counts.map(([label, count]) => (
            <div key={label} className="flex gap-1">
              <dt className="text-zinc-600">{label}:</dt>
              <dd className="font-medium">{count}</dd>
            </div>
          ))}
        </dl>
        <p className="text-xs text-zinc-600">
          Graph coverage is limited to indexed, source-verified facts; missing
          links do not establish that a relationship is absent.
        </p>
      </section>
    );
  };

  const renderStatusResult = (message: Message): ReactNode | null => {
    const result = message.assistantResult;
    if (!result) return null;
    const failure = failureMessage(result);
    const statusLabels: Partial<Record<AssistantResultType, string>> = {
      clarification: "More information needed",
      approval_required: "Approval required",
      approval_invalidated: "Approval needs review",
      routing_unavailable: "Request could not be routed",
      tool_error: "Research step failed",
      action_outcome_unknown: "Action status needs checking",
      unavailable: "Feature unavailable",
      discovery_unavailable: "Paper discovery unavailable",
      graph_index_approval_required: "Graph indexing needs approval",
      graph_index_approval_invalid:
        "Graph indexing approval expired or changed",
      graph_indexing_unavailable: "Graph indexing unavailable",
      note_not_found: "Note unavailable",
      note_source_unavailable: "Note source could not be verified",
      note_version_conflict: "Note changed",
      note_update_rejected: "Note update not applied",
      report_evidence_unavailable: "Report needs more evidence",
      research_evidence_unavailable: "Research evidence unavailable",
      gap_analysis_evidence_unavailable: "Gap-analysis evidence unavailable",
      experiment_evidence_unavailable: "Experiment evidence unavailable",
      comparison_scope_unavailable: "Comparison needs a valid paper selection",
      claim_scope_unavailable:
        "Claim verification needs a valid paper selection",
      source_selection_unavailable: "Selected passage is no longer available",
      visual_selection_required: "Select a figure or crop first",
      visual_analysis_unavailable: "Figure analysis unavailable",
    };
    const label = statusLabels[result.result_type as AssistantResultType];
    if (!label) return null;
    const missing =
      result.result_type === "clarification" &&
      isRecord(result.structured_payload) &&
      Array.isArray(result.structured_payload.missing_information)
        ? result.structured_payload.missing_information.flatMap(
            (item): string[] => {
              if (item === "ready_paper") return ["Select a ready paper."];
              if (item === "select_or_identify_paper") {
                return ["Select a paper or clarify its title."];
              }
              if (
                [
                  "paper",
                  "paper_scope",
                  "selected_paper_ids",
                  "scope",
                ].includes(String(item))
              ) {
                return ["Choose the paper or project scope."];
              }
              return [];
            },
          )
        : [];
    return (
      <section
        className="rounded border border-amber-200 bg-amber-50 p-3 text-amber-950"
        aria-label={label}
      >
        <p className="font-semibold">{failure?.heading ?? label}</p>
        {failure && <p className="mt-1 text-xs">{failure.detail}</p>}
        {missing.length > 0 && (
          <ul className="mt-1 list-disc pl-5 text-xs">
            {missing.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        )}
        {!failure && (
          <div className="mt-1">
            {renderContent(message.content, message.citations)}
          </div>
        )}
      </section>
    );
  };

  const resultCardRenderers: Partial<
    Record<AssistantResultType, (message: Message) => ReactNode | null>
  > = {
    comparison: renderComparison,
    discovery_results: renderDiscovery,
    discovery_unavailable: renderStatusResult,
    visual_analysis: renderVisualAnalysis,
    reading_brief: renderReadingBrief,
    claim_verification: renderClaimVerification,
    notes_list: renderNotes,
    note: renderNotes,
    research_report: renderReport,
    graph_index_jobs: renderGraphIndex,
    research_draft: (message) =>
      renderResearchDiscovery(message) ??
      renderContent(message.content, message.citations),
    unavailable: renderStatusResult,
    clarification: renderStatusResult,
    approval_required: renderStatusResult,
    approval_invalidated: renderStatusResult,
    routing_unavailable: renderStatusResult,
    tool_error: renderStatusResult,
    action_outcome_unknown: renderStatusResult,
    graph_index_approval_required: renderStatusResult,
    graph_index_approval_invalid: renderStatusResult,
    graph_indexing_unavailable: renderStatusResult,
    note_not_found: renderStatusResult,
    note_source_unavailable: renderStatusResult,
    note_version_conflict: renderStatusResult,
    note_update_rejected: renderStatusResult,
    report_evidence_unavailable: renderStatusResult,
    research_evidence_unavailable: renderStatusResult,
    gap_analysis_evidence_unavailable: renderStatusResult,
    experiment_evidence_unavailable: renderStatusResult,
    comparison_scope_unavailable: renderStatusResult,
    claim_scope_unavailable: renderStatusResult,
    source_selection_unavailable: renderStatusResult,
    visual_selection_required: renderStatusResult,
    visual_analysis_unavailable: renderStatusResult,
  };

  const renderResultCard = (message: Message): ReactNode | null => {
    const type = message.assistantResult?.result_type;
    if (!type || !Object.hasOwn(resultCardRenderers, type)) return null;
    return resultCardRenderers[type as AssistantResultType]?.(message) ?? null;
  };

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-xs">
      <div className="border-b border-zinc-200 px-4 py-3">
        <h2 className="text-sm font-semibold text-zinc-900">
          Research Assistant
        </h2>
        <p className="text-xs text-zinc-500">
          Ask naturally to read, compare, verify, or question your research.
          Click any [n] chip to jump to evidence.
        </p>
        {routedIntent && (
          <p className="mt-2 text-xs text-blue-700" role="status">
            Using: {intentLabel(routedIntent)}
          </p>
        )}
      </div>

      {conversations &&
        onSelectConversation &&
        onCreateConversation &&
        onRenameConversation &&
        onDeleteConversation && (
          <ConversationNavigator
            conversations={conversations}
            activeConversation={activeConversation || null}
            onSelectConversation={onSelectConversation}
            onCreateConversation={onCreateConversation}
            onRenameConversation={onRenameConversation}
            onArchiveConversation={onArchiveConversation}
            onDeleteConversation={onDeleteConversation}
            disabled={disabled}
          />
        )}

      {/* Error banner */}
      {error && (
        <div className="flex items-center justify-between border-b border-red-200 bg-red-50 px-4 py-2 text-xs text-red-800">
          <span>{error}</span>
          {onDismissError && (
            <button
              type="button"
              onClick={onDismissError}
              className="text-red-500 hover:text-red-700 cursor-pointer ml-2"
              title="Dismiss error"
              aria-label="Dismiss error"
            >
              ✕
            </button>
          )}
        </div>
      )}

      {activeRun && (
        <section
          className="border-b border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-700"
          aria-label="Active research run"
        >
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="font-medium">
                {activeRun.action_summary ||
                  activeRun.intent ||
                  "Research request"}
              </p>
              <p role="status">{runStatusLabel(activeRun.status)}</p>
              {(activeRun.status === "QUEUED" ||
                activeRun.status === "RUNNING") && (
                <p className="mt-1 text-zinc-500">
                  This request is saved. You can leave and return to check its
                  status.
                </p>
              )}
              {activeRun.status === "CANCELLED" && (
                <p className="mt-1 text-zinc-500">
                  No result was marked complete.
                </p>
              )}
              {activeRun.status === "NEEDS_INPUT" && (
                <p className="mt-1 text-zinc-500">
                  Answer the question below so mYrA can continue.
                </p>
              )}
              {activeRun.status === "AWAITING_APPROVAL" && (
                <p className="mt-1 text-zinc-500">
                  Nothing will change until you approve the proposed action.
                </p>
              )}
            </div>
            {(activeRun.status === "QUEUED" ||
              activeRun.status === "RUNNING") &&
              onCancelRun && (
                <button
                  type="button"
                  onClick={() => void onCancelRun(activeRun.id)}
                  className="rounded border border-zinc-300 px-2 py-1 hover:bg-white"
                >
                  Cancel run
                </button>
              )}
          </div>
          {activeRun.status === "NEEDS_INPUT" && onResumeRun && (
            <form
              className="mt-3 flex gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                if (resumeInput.trim()) {
                  void onResumeRun(activeRun.id, resumeInput.trim());
                  setResumeInput("");
                }
              }}
            >
              <input
                aria-label="Clarification"
                value={resumeInput}
                onChange={(event) => setResumeInput(event.target.value)}
                placeholder="Answer the clarification…"
                className="min-w-0 flex-1 rounded border border-zinc-300 px-2 py-1.5"
              />
              <button
                type="submit"
                disabled={!resumeInput.trim()}
                className="rounded bg-zinc-900 px-3 py-1.5 text-white disabled:opacity-40"
              >
                Continue
              </button>
            </form>
          )}
          {activeRun.status === "AWAITING_APPROVAL" &&
            pendingAction &&
            onDecideAction && (
              <div className="mt-3 rounded border border-amber-300 bg-amber-50 p-3">
                <p className="font-medium">
                  Review proposed action:{" "}
                  {pendingAction.action_type.replaceAll("_", " ")}
                </p>
                <p className="mt-1 text-xs">
                  Check the exact proposal below. The action will not run unless
                  you approve it.
                </p>
                <pre className="mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-words text-[11px]">
                  {JSON.stringify(pendingAction.arguments, null, 2)}
                </pre>
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    onClick={() => void onDecideAction(pendingAction.id, true)}
                    className="rounded bg-emerald-700 px-3 py-1.5 text-white"
                  >
                    Approve
                  </button>
                  <button
                    type="button"
                    onClick={() => void onDecideAction(pendingAction.id, false)}
                    className="rounded border border-zinc-300 px-3 py-1.5"
                  >
                    Reject
                  </button>
                </div>
              </div>
            )}
        </section>
      )}

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center text-center text-zinc-400 text-sm">
            {activeConversation
              ? "Ask a question over the indexed papers in this project."
              : "Click '+ New Chat' to begin asking questions."}
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex flex-col ${msg.role === "USER" ? "items-end" : "items-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${
                  msg.role === "USER"
                    ? "bg-zinc-900 text-white"
                    : "bg-zinc-100 text-zinc-900 border border-zinc-200"
                }`}
              >
                {msg.role === "ASSISTANT"
                  ? (renderResultCard(msg) ??
                    renderContent(msg.content, msg.citations))
                  : msg.content}
              </div>
              <span className="mt-1 text-[10px] text-zinc-400">
                {msg.role === "USER" ? "You" : "mYrA"}
              </span>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex items-start">
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-2.5 text-sm text-zinc-600 animate-pulse">
              {routedIntent
                ? `${intentLabel(routedIntent)} in progress…`
                : "Routing your research request…"}
            </div>
          </div>
        )}
      </div>

      {/* Input bar */}
      <form onSubmit={handleSubmit} className="border-t border-zinc-200 p-3">
        {sourceSelection && (
          <div className="mb-2 flex items-start justify-between gap-2 rounded border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
            <span>
              Selected passage from page {sourceSelection.page_number}. Your
              next question will be checked against this exact text.
            </span>
            {onClearSourceSelection && (
              <button
                type="button"
                onClick={onClearSourceSelection}
                className="shrink-0 underline"
              >
                Clear
              </button>
            )}
          </div>
        )}
        {visualSelection && (
          <div className="mb-2 flex items-start justify-between gap-2 rounded border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
            <span>
              Selected visual crop from page {visualSelection.page_number}. It
              will be sent to DeepSeek for figure analysis, not treated as a
              text citation.
            </span>
            {onClearVisualSelection && (
              <button
                type="button"
                onClick={onClearVisualSelection}
                className="shrink-0 underline"
              >
                Clear
              </button>
            )}
          </div>
        )}
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={disabled || isLoading}
            placeholder={
              disabled
                ? !activeConversation
                  ? "Create a chat to begin asking questions…"
                  : "Upload and index a paper to begin asking questions…"
                : "Ask a grounded research question…"
            }
            className="flex-1 rounded-lg border border-zinc-300 px-3 py-2 text-sm text-zinc-900 placeholder-zinc-400 focus:border-zinc-900 focus:outline-hidden disabled:bg-zinc-50 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading || disabled}
            className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-40 cursor-pointer"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}

function remarkCitationLinks() {
  return (tree: Root) => {
    visit(tree, "text", (node, index, parent) => {
      if (index === undefined || !parent || !/\[\d+\]/.test(node.value)) return;
      const children: PhrasingContent[] = node.value
        .split(/(\[\d+\])/g)
        .filter(Boolean)
        .map((part) => {
          const match = part.match(/^\[(\d+)\]$/);
          return match
            ? {
                type: "link",
                url: `myra-citation:${match[1]}`,
                children: [{ type: "text", value: part }],
              }
            : { type: "text", value: part };
        });
      parent.children.splice(index, 1, ...children);
      return index + children.length;
    });
  };
}

function intentLabel(intent: AssistantIntent): string {
  const labels: Record<AssistantIntent, string> = {
    help: "Help",
    qa: "Question answering",
    read_paper: "Paper reading",
    compare: "Paper comparison",
    verify_claim: "Claim verification",
    discover: "Paper discovery",
    notes: "Research notes",
    report: "Report drafting",
    research: "Research workflow",
    gap_analysis: "Gap analysis",
    experiment_plan: "Experiment planning",
    translate: "PDF translation",
    vision: "Figure understanding",
    graph: "Graph exploration",
    clarify: "Clarification",
  };
  return labels[intent];
}
