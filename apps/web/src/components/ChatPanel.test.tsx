import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ChatPanel from "./ChatPanel";
import type { Citation, Message } from "@/types";

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
});
