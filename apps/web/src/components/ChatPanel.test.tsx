import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ChatPanel from "./ChatPanel";
import type {
  AssistantRunResponse,
  AssistantRunResult,
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
});
