import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ChatPanel from "./ChatPanel";
import type {
  AssistantRunResponse,
  AssistantRunResult,
  AssistantApprovalResponse,
  Citation,
  Message,
} from "@/types";

const mockCitation: Citation = {
  citation_index: 1,
  evidence_id: "E1",
  paper_id: "paper-123",
  page_number: 1,
  quote: "Transformers achieve state of the art results.",
  bounding_boxes: [],
};

const mockMessages: Message[] = [
  {
    id: "msg-1",
    conversation_id: "conv-1",
    role: "USER",
    content: "What does the paper propose?",
    citations: [],
    evidence: [],
    created_at: new Date().toISOString(),
  },
  {
    id: "msg-2",
    conversation_id: "conv-1",
    role: "ASSISTANT",
    content: "The paper proposes the Transformer architecture [1].",
    citations: [mockCitation],
    evidence: [],
    created_at: new Date().toISOString(),
  },
];

const comparisonResult: AssistantRunResult = {
  result_type: "comparison",
  display_text: "Comparison summary [1].",
  structured_payload: {
    matrix: {
      paper_ids: ["paper-123", "paper-456"],
      dimensions: ["method", "results"],
      cells: [
        {
          paper_id: "paper-123",
          dimension: "method",
          status: "evidence_available",
          excerpts: [
            {
              evidence: {
                id: "E1",
                paper_id: "paper-123",
                paper_title: "Transformer Paper",
                quote: "Transformers achieve state of the art results.",
              },
              citation: mockCitation,
            },
          ],
        },
        {
          paper_id: "paper-123",
          dimension: "results",
          status: "not_found",
          message: "Not reported in retrieved evidence.",
          excerpts: [],
        },
      ],
    },
    synthesis: {},
  },
  citations: [mockCitation],
  warnings: ["Candidate source excerpts only."],
  usage: {},
  available_actions: [],
  artifact_ids: [],
};

describe("ChatPanel", () => {
  it("renders messages and citation chips", () => {
    render(
      <ChatPanel
        messages={mockMessages}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByText("What does the paper propose?"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The paper proposes the Transformer architecture/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "[1]" })).toBeInTheDocument();
  });

  it("renders visual readings as interpretation, separate from text citations", () => {
    const onVisualSourceClick = vi.fn();
    const visualMessage: Message = {
      id: "visual-result",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "The series rises.",
      citations: [],
      evidence: [],
      assistantResult: {
        result_type: "visual_analysis",
        display_text: "The series rises.",
        structured_payload: {
          analysis: {
            observations: [{ statement: "The blue line rises." }],
            readings: [
              {
                label: "At x=2",
                value: "about 4",
                unit: null,
                kind: "plot_estimate",
              },
            ],
            interpretation: "The trend is upward.",
            uncertainty_notes: ["The y-axis labels are blurred."],
          },
          visual_source: {
            source_kind: "visual",
            project_id: "project-1",
            paper_id: "paper-123",
            document_sha256: "a".repeat(64),
            page_number: 4,
            crop_sha256: "b".repeat(64),
            crop_box_normalized_top_left: {
              left: 0.1,
              top: 0.2,
              right: 0.8,
              bottom: 0.9,
            },
            text_citation: false,
          },
          cache_status: "miss",
          requested_model: "deepseek-flash",
        },
        citations: [],
        warnings: [],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      },
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[visualMessage]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        onVisualSourceClick={onVisualSourceClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText(/not a text quotation/)).toBeInTheDocument();
    expect(screen.getByText(/estimated from plot/)).toBeInTheDocument();
    expect(
      screen.getByText("The y-axis labels are blurred."),
    ).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Open original page 4" }),
    );
    expect(onVisualSourceClick).toHaveBeenCalledWith(
      expect.objectContaining({ source_kind: "visual", paper_id: "paper-123" }),
    );
  });

  it("offers bounded discovery from a research evidence gap", async () => {
    const onSendMessage = vi.fn().mockResolvedValue(undefined);
    const message: Message = {
      id: "research-gap",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "The selected papers do not report this evaluation [1].",
      citations: [mockCitation],
      evidence: [],
      created_at: new Date().toISOString(),
      assistantResult: {
        result_type: "research_draft",
        display_text: "The selected papers do not report this evaluation [1].",
        structured_payload: {
          discovery_query: "small-data evaluation methods",
        },
        citations: [mockCitation],
        warnings: [],
        usage: {},
        available_actions: ["discover"],
        artifact_ids: [],
      },
    };
    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={onSendMessage}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Discover papers for this gap" }),
    );
    expect(
      screen.getByText(/selected papers do not report this evaluation/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "[1]" })).toBeInTheDocument();
    await waitFor(() =>
      expect(onSendMessage).toHaveBeenCalledWith(
        "Find academic papers about: small-data evaluation methods",
      ),
    );
  });

  it("renders tables and math, keeps citations interactive, and ignores raw HTML", () => {
    const onCitationClick = vi.fn();
    const message: Message = {
      id: "markdown-answer",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content:
        "**Supported result** [1]\n\n| Metric | Value |\n| --- | ---: |\n| Accuracy | 92% |\n\nInline $x^2$ and display:\n\n$$y = mx + b$$\n\n<script>alert('unsafe')</script>",
      citations: [mockCitation],
      evidence: [],
      created_at: new Date().toISOString(),
    };
    const { container } = render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText("Supported result").tagName).toBe("STRONG");
    expect(container.querySelector("table")).toBeInTheDocument();
    expect(container.querySelectorAll(".katex").length).toBeGreaterThan(0);
    expect(container.querySelector("script")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "[1]" }));
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("calls onCitationClick when citation chip is clicked", () => {
    const onCitationClick = vi.fn();
    render(
      <ChatPanel
        messages={mockMessages}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    const chip = screen.getByRole("button", { name: "[1]" });
    fireEvent.click(chip);
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("renders comparison cells and opens exact source citations", () => {
    const onCitationClick = vi.fn();
    const message: Message = {
      id: "comparison-message",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: comparisonResult.display_text,
      citations: [mockCitation],
      evidence: [],
      assistantResult: comparisonResult,
      created_at: new Date().toISOString(),
    };
    const { container } = render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText("Transformer Paper")).toBeInTheDocument();
    expect(screen.getByText("method")).toBeInTheDocument();
    expect(
      screen.getByText("Transformers achieve state of the art results."),
    ).toBeInTheDocument();
    expect(container.querySelector("table")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Page 1/ }));
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("renders per-paper comparison findings and their citations", () => {
    const onCitationClick = vi.fn();
    const result: AssistantRunResult = {
      ...comparisonResult,
      display_text: "Per-paper findings",
      structured_payload: {
        version: 2,
        question: "What method is used?",
        paper_findings: [
          {
            paper_id: "paper-123",
            title: "Transformer Paper",
            summary: "The paper uses attention. [1]",
            evidence_available: true,
            citations: [mockCitation],
            limitations: [],
          },
        ],
      },
    };
    const message: Message = {
      id: "simple-comparison",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: result.display_text,
      citations: [mockCitation],
      evidence: [],
      assistantResult: result,
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByRole("region", { name: "Paper comparison" }),
    ).toBeInTheDocument();
    expect(screen.getByText("The paper uses attention.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "[1]" }));
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("shows benchmark comparability and opens sources from both papers", () => {
    const secondCitation: Citation = {
      ...mockCitation,
      citation_index: 2,
      evidence_id: "E2",
      paper_id: "paper-456",
      page_number: 3,
      quote: "The reported result is 88 percent.",
    };
    const result: AssistantRunResult = {
      ...comparisonResult,
      citations: [mockCitation, secondCitation],
      structured_payload: {
        ...comparisonResult.structured_payload,
        synthesis: {
          benchmark_comparisons: [
            {
              left_paper_id: "paper-123",
              right_paper_id: "paper-456",
              left_result: "90 percent",
              right_result: "88 percent",
              evidence_ids: ["E1", "E2"],
              comparability: {
                status: "not directly comparable",
                reasons: ["split differs between papers"],
              },
            },
          ],
        },
      },
    };
    const onCitationClick = vi.fn();
    const message: Message = {
      id: "benchmark-comparison",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: result.display_text,
      citations: result.citations,
      evidence: [],
      assistantResult: result,
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText("Not directly comparable.")).toBeInTheDocument();
    expect(
      screen.getByText("split differs between papers"),
    ).toBeInTheDocument();
    const section = screen.getByRole("region", {
      name: "Benchmark comparisons",
    });
    fireEvent.click(
      within(section).getByRole("button", { name: "Page 1 ·[1]" }),
    );
    fireEvent.click(
      within(section).getByRole("button", { name: "Page 3 ·[2]" }),
    );
    expect(onCitationClick.mock.calls).toEqual([
      [mockCitation],
      [secondCitation],
    ]);
  });

  it("shows missing cells and the candidate-only warning", () => {
    const message: Message = {
      id: "comparison-missing",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: comparisonResult.display_text,
      citations: [mockCitation],
      evidence: [],
      assistantResult: comparisonResult,
      created_at: new Date().toISOString(),
    };
    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByRole("note")).toHaveTextContent(
      /Candidate source excerpts only/,
    );
    expect(
      screen.getAllByText("Not reported in retrieved evidence.").length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("paper-456")).toBeInTheDocument();
  });

  it("falls back to the regular answer when the comparison matrix is malformed", () => {
    const message: Message = {
      id: "comparison-malformed",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "A readable fallback answer [1].",
      citations: [mockCitation],
      evidence: [],
      assistantResult: {
        ...comparisonResult,
        structured_payload: { matrix: { paper_ids: "not-an-array" } },
      },
      created_at: new Date().toISOString(),
    };
    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText(/A readable fallback answer/)).toBeInTheDocument();
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  it("renders discovery cards as metadata only and rejects unsafe catalog links", () => {
    const result: AssistantRunResult = {
      result_type: "discovery_results",
      display_text: "Found two catalog records.",
      structured_payload: {
        metadata_only: true,
        items: [
          {
            catalog: "arxiv",
            catalog_id: "2401.12345",
            title: "A Research Paper",
            authors: ["Researcher"],
            publication_year: 2024,
            arxiv_id: "2401.12345",
            abstract: "A short catalog abstract.",
            source_url: "https://arxiv.org/abs/2401.12345",
            open_access: true,
            possible_duplicate: true,
          },
          {
            catalog: "openalex",
            catalog_id: "W2",
            title: "Unsafe candidate",
            source_url: "javascript:alert(1)",
          },
        ],
        source_errors: { openalex: "TimeoutException" },
      },
      citations: [],
      warnings: [],
      usage: {},
      available_actions: [],
      artifact_ids: [],
    };
    const message: Message = {
      id: "discovery-message",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: result.display_text,
      citations: [],
      evidence: [],
      assistantResult: result,
      created_at: new Date().toISOString(),
    };

    const { container } = render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByRole("note")).toHaveTextContent(/Metadata only/);
    expect(screen.getByText("A Research Paper")).toBeInTheDocument();
    expect(screen.getByText(/Possible title match/)).toBeInTheDocument();
    expect(
      screen.getByText(/openalex search was unavailable/),
    ).toBeInTheDocument();
    expect(
      container.querySelectorAll('a[href^="https://arxiv.org/"]'),
    ).toHaveLength(1);
    expect(screen.queryByText("Unsafe candidate")).not.toBeInTheDocument();
  });

  it("requires a separate approval click before importing a discovery result", async () => {
    const candidate = {
      catalog: "arxiv",
      catalog_id: "2401.12345v1",
      title: "A Research Paper",
      authors: [],
      publication_year: 2024,
      doi: null,
      arxiv_id: "2401.12345v1",
      abstract: "Catalog description.",
      source_url: "https://arxiv.org/abs/2401.12345v1",
      pdf_url: "https://arxiv.org/pdf/2401.12345v1",
      open_access: true,
      possible_duplicate: false,
    };
    const pending: AssistantApprovalResponse = {
      id: "action-1",
      run_id: "run-1",
      action_type: "discovery_import",
      arguments: { candidate },
      source_fingerprint: "a".repeat(64),
      status: "PENDING",
      expires_at: new Date(Date.now() + 60_000).toISOString(),
      decided_at: null,
    };
    const onProposeDiscoveryImport = vi.fn().mockResolvedValue(pending);
    const onDecideDiscoveryImport = vi.fn().mockResolvedValue({
      ...pending,
      status: "APPROVED",
      import_result: {
        paper_id: "paper-imported",
        job_id: "job-imported",
        status: "PROCESSING",
      },
    });
    const message: Message = {
      id: "discovery-import-message",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "Found a candidate.",
      citations: [],
      evidence: [],
      assistantResult: {
        result_type: "discovery_results",
        display_text: "Found a candidate.",
        structured_payload: {
          run_id: "run-1",
          items: [candidate],
          source_errors: {},
          metadata_only: true,
        },
        citations: [],
        warnings: [],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      },
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
        onProposeDiscoveryImport={onProposeDiscoveryImport}
        onDecideDiscoveryImport={onDecideDiscoveryImport}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Review import" }));
    await waitFor(() =>
      expect(
        screen.getByText(/Confirm download from arxiv.org/),
      ).toBeInTheDocument(),
    );
    expect(onProposeDiscoveryImport).toHaveBeenCalledWith("run-1", candidate);
    fireEvent.click(screen.getByRole("button", { name: "Approve and import" }));
    await waitFor(() =>
      expect(
        screen.getByText("Approved; paper queued for ingestion."),
      ).toBeInTheDocument(),
    );
    expect(onDecideDiscoveryImport).toHaveBeenCalledWith("action-1", true);
  });

  it("keeps normal QA answers in their existing Markdown rendering", () => {
    const { container } = render(
      <ChatPanel
        messages={[mockMessages[1]]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByText(/The paper proposes the Transformer architecture/),
    ).toBeInTheDocument();
    expect(container.querySelector("table")).toBeNull();
  });

  it("calls onSendMessage when user submits input", () => {
    const onSendMessage = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        isLoading={false}
        onSendMessage={onSendMessage}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    const input = screen.getByPlaceholderText(
      /Ask a grounded research question/,
    );
    fireEvent.change(input, { target: { value: "How many layers?" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(onSendMessage).toHaveBeenCalledWith("How many layers?");
  });

  it("shows the selected passage scope and lets the user clear it", () => {
    const onClearSourceSelection = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
        sourceSelection={{
          paper_id: "paper-123",
          page_number: 4,
          quote: "A selected passage.",
          document_sha256: "aa".repeat(32),
        }}
        onClearSourceSelection={onClearSourceSelection}
      />,
    );

    expect(
      screen.getByText(/Selected passage from page 4/),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear" }));
    expect(onClearSourceSelection).toHaveBeenCalled();
  });

  it("renders error banner and calls onDismissError when dismissed", () => {
    const onDismiss = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        isLoading={false}
        error="Failed to connect to DeepSeek"
        onDismissError={onDismiss}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByText("Failed to connect to DeepSeek"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Dismiss error" }));
    expect(onDismiss).toHaveBeenCalled();
  });

  it("offers clarification, approval, and cancellation controls for persisted runs", () => {
    const onResumeRun = vi.fn();
    const onDecideAction = vi.fn();
    const onCancelRun = vi.fn();
    const run: AssistantRunResponse = {
      id: "run-42",
      project_id: "project-1",
      conversation_id: "conv-1",
      status: "AWAITING_APPROVAL",
      intent: "discover",
      action_summary: "Import a paper",
      stage: "assistant.approval",
      result: null,
      safe_error: null,
      usage: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    const { rerender } = render(
      <ChatPanel
        messages={[]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
        activeRun={{ ...run, status: "RUNNING" }}
        onCancelRun={onCancelRun}
        onResumeRun={onResumeRun}
        onDecideAction={onDecideAction}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Cancel run" }));
    expect(onCancelRun).toHaveBeenCalledWith("run-42");

    rerender(
      <ChatPanel
        messages={[]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
        activeRun={{ ...run, status: "NEEDS_INPUT" }}
        onCancelRun={onCancelRun}
        onResumeRun={onResumeRun}
        onDecideAction={onDecideAction}
      />,
    );
    fireEvent.change(screen.getByLabelText("Clarification"), {
      target: { value: "Compare two selected papers" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    expect(onResumeRun).toHaveBeenCalledWith(
      "run-42",
      "Compare two selected papers",
    );

    rerender(
      <ChatPanel
        messages={[]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
        activeRun={run}
        pendingAction={{
          id: "action-9",
          run_id: "run-42",
          action_type: "import_paper",
          arguments: { paper_id: "paper-1" },
          source_fingerprint: "a".repeat(64),
          status: "PENDING",
          expires_at: new Date().toISOString(),
          decided_at: null,
        }}
        onCancelRun={onCancelRun}
        onResumeRun={onResumeRun}
        onDecideAction={onDecideAction}
      />,
    );
    expect(screen.getByText(/paper_id/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(onDecideAction).toHaveBeenCalledWith("action-9", true);
    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(onDecideAction).toHaveBeenCalledWith("action-9", false);
  });

  it("renders a typed reading brief and keeps section citations interactive", () => {
    const onCitationClick = vi.fn();
    const message: Message = {
      id: "reading-brief",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "Brief [1].",
      citations: [mockCitation],
      evidence: [],
      assistantResult: {
        result_type: "reading_brief",
        display_text: "Brief [1].",
        structured_payload: {
          sections: [
            {
              title: "Method",
              content: "The paper uses attention [1].",
              citation_indexes: [1],
            },
          ],
        },
        citations: [mockCitation],
        warnings: [],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      },
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByRole("region", { name: "Paper reading brief" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Method")).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Source [1] · page 1" }),
    );
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("renders claim verification as a scoped verdict with source navigation", () => {
    const onCitationClick = vi.fn();
    const message: Message = {
      id: "claim-verification",
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "Verdict: supported [1].",
      citations: [mockCitation],
      evidence: [],
      assistantResult: {
        result_type: "claim_verification",
        display_text: "Verdict: supported [1].",
        structured_payload: {
          claim: "Attention helps model dependencies.",
          verdict: "supported",
          explanation: "A current passage supports it.",
          scope_note: "Limited to selected papers.",
        },
        citations: [mockCitation],
        warnings: [],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      },
      created_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[message]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByRole("region", { name: "Claim verification" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Limited to selected papers/)).toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Source [1] · page 1" }),
    );
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("renders notes, reports, and graph indexing outcomes with typed cards", () => {
    const baseMessage: Omit<Message, "assistantResult" | "id"> = {
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "Result text.",
      citations: [],
      evidence: [],
      created_at: new Date().toISOString(),
    };
    const messages: Message[] = [
      {
        ...baseMessage,
        id: "notes-list",
        assistantResult: {
          result_type: "notes_list",
          display_text: "One active note.",
          structured_payload: {
            total: 1,
            items: [
              {
                id: "note-1",
                title: "Dataset decision",
                content: "Use the small-data split.",
                type: "DECISION",
                status: "ACTIVE",
                version: 2,
                sources: [{ page_number: 2 }],
              },
            ],
          },
          citations: [],
          warnings: [],
          usage: {},
          available_actions: [],
          artifact_ids: [],
        },
      },
      {
        ...baseMessage,
        id: "report",
        citations: [mockCitation],
        assistantResult: {
          result_type: "research_report",
          display_text: "Report draft.",
          structured_payload: {
            report_markdown: "## Findings\n\nA reported result [1].",
            scope: "selection",
            saved: false,
            source_manifest: [
              { citation_index: 1, page_number: 1, paper_title: "Paper" },
            ],
          },
          citations: [mockCitation],
          warnings: [],
          usage: {},
          available_actions: [],
          artifact_ids: [],
        },
      },
      {
        ...baseMessage,
        id: "graph-index",
        assistantResult: {
          result_type: "graph_index_jobs",
          display_text: "Queued graph indexing for 1 paper.",
          structured_payload: { queued_count: 1, skipped_count: 0 },
          citations: [],
          warnings: [],
          usage: {},
          available_actions: [],
          artifact_ids: [],
        },
      },
    ];
    const onCitationClick = vi.fn();

    render(
      <ChatPanel
        messages={messages}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText("Dataset decision")).toBeInTheDocument();
    expect(screen.getByText("Use the small-data split.")).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Research report" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Draft report — not saved/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Paper · page 1" }));
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
    expect(
      screen.getByRole("region", { name: "Graph indexing jobs" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Graph coverage is limited to indexed/),
    ).toBeInTheDocument();
  });

  it("uses a safe text fallback for malformed known and unknown result types", () => {
    const unknownResult = {
      result_type: "future_card_type",
      display_text: "Future result.",
      structured_payload: { arbitrary: true },
      citations: [mockCitation],
      warnings: [],
      usage: {},
      available_actions: [],
      artifact_ids: [],
    } as unknown as AssistantRunResult;
    const messages: Message[] = [
      {
        id: "malformed-brief",
        conversation_id: "conv-1",
        role: "ASSISTANT",
        content: "Malformed card falls back [1].",
        citations: [mockCitation],
        evidence: [],
        assistantResult: {
          ...unknownResult,
          result_type: "reading_brief",
          structured_payload: { sections: "not-an-array" },
        } as unknown as AssistantRunResult,
        created_at: new Date().toISOString(),
      },
      {
        id: "unknown-result",
        conversation_id: "conv-1",
        role: "ASSISTANT",
        content: "Unknown card falls back [1].",
        citations: [mockCitation],
        evidence: [],
        assistantResult: unknownResult,
        created_at: new Date().toISOString(),
      },
    ];
    const onCitationClick = vi.fn();

    render(
      <ChatPanel
        messages={messages}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={onCitationClick}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText(/Malformed card falls back/)).toBeInTheDocument();
    expect(screen.getByText(/Unknown card falls back/)).toBeInTheDocument();
    const citationButtons = screen.getAllByRole("button", { name: "[1]" });
    fireEvent.click(citationButtons[1]);
    expect(onCitationClick).toHaveBeenCalledWith(mockCitation);
  });

  it("labels clarification and unavailable outcomes instead of implying success", () => {
    const message = (
      id: string,
      resultType: "clarification" | "graph_indexing_unavailable",
      payload: Record<string, unknown>,
      content: string,
    ): Message => ({
      id,
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content,
      citations: [],
      evidence: [],
      assistantResult: {
        result_type: resultType,
        display_text: content,
        structured_payload: payload,
        citations: [],
        warnings: [],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      },
      created_at: new Date().toISOString(),
    });

    render(
      <ChatPanel
        messages={[
          message(
            "clarify",
            "clarification",
            { missing_information: ["paper_scope"] },
            "Choose a paper.",
          ),
          message(
            "graph-offline",
            "graph_indexing_unavailable",
            {},
            "Graph indexing is offline; ordinary research still works.",
          ),
        ]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(
      screen.getByRole("region", { name: "More information needed" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Choose the paper or project scope."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("region", { name: "Graph indexing unavailable" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/ordinary research still works/),
    ).toBeInTheDocument();
  });

  it("explains known provider, budget, and changed-source failures without exposing payload details", () => {
    const makeFailure = (
      id: string,
      resultType: AssistantRunResult["result_type"],
      warning: string,
    ): Message => ({
      id,
      conversation_id: "conv-1",
      role: "ASSISTANT",
      content: "Internal detail: /srv/private/config.toml",
      citations: [],
      evidence: [],
      assistantResult: {
        result_type: resultType,
        display_text: "Internal detail: /srv/private/config.toml",
        structured_payload: {},
        citations: [],
        warnings: [warning],
        usage: {},
        available_actions: [],
        artifact_ids: [],
      } as AssistantRunResult,
      created_at: new Date().toISOString(),
    });

    render(
      <ChatPanel
        messages={[
          makeFailure(
            "provider",
            "visual_analysis_unavailable",
            "PROVIDER_UNAVAILABLE",
          ),
          makeFailure("budget", "visual_analysis_unavailable", "BUDGET_DENIED"),
          makeFailure(
            "source",
            "approval_invalidated",
            "APPROVAL_SOURCE_CHANGED",
          ),
        ]}
        isLoading={false}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByText("Research model unavailable")).toBeInTheDocument();
    expect(screen.getByText("Spending limit reached")).toBeInTheDocument();
    expect(screen.getByText("The source changed")).toBeInTheDocument();
    expect(screen.queryByText(/private\/config\.toml/)).not.toBeInTheDocument();
  });

  it("uses plain language for saved pending runs and does not show raw stages", () => {
    const activeRun: AssistantRunResponse = {
      id: "run-1",
      project_id: "project-1",
      conversation_id: "conv-1",
      status: "RUNNING",
      intent: "qa",
      action_summary: "Answer a paper question",
      stage: "assistant.tool.qa.internal",
      result: null,
      safe_error: null,
      usage: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };

    render(
      <ChatPanel
        messages={[]}
        isLoading={false}
        activeRun={activeRun}
        onSendMessage={vi.fn()}
        onCitationClick={vi.fn()}
        activeCitation={null}
        disabled={false}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent(
      "Research in progress",
    );
    expect(screen.getByText(/request is saved/i)).toBeInTheDocument();
    expect(
      screen.queryByText(/assistant\.tool\.qa\.internal/),
    ).not.toBeInTheDocument();
  });
});
