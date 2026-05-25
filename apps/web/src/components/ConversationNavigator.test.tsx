import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ConversationNavigator from "./ConversationNavigator";
import type { Conversation } from "@/types";

const mockConversations: Conversation[] = [
  {
    id: "conv-1",
    project_id: "proj-1",
    title: "Initial Chat",
    message_count: 3,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
  {
    id: "conv-2",
    project_id: "proj-1",
    title: "Follow-up Chat",
    message_count: 0,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  },
];

describe("ConversationNavigator", () => {
  it("renders conversation options in select", () => {
    render(
      <ConversationNavigator
        conversations={mockConversations}
        activeConversation={mockConversations[0]}
        onSelectConversation={vi.fn()}
        onCreateConversation={vi.fn()}
        onRenameConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("combobox", { name: /Chat:/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Initial Chat/)).toBeInTheDocument();
    expect(screen.getByText(/Follow-up Chat/)).toBeInTheDocument();
  });

  it("calls onSelectConversation when conversation changed in select", () => {
    const onSelect = vi.fn();
    render(
      <ConversationNavigator
        conversations={mockConversations}
        activeConversation={mockConversations[0]}
        onSelectConversation={onSelect}
        onCreateConversation={vi.fn()}
        onRenameConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
      />,
    );

    const select = screen.getByRole("combobox", { name: /Chat:/i });
    fireEvent.change(select, { target: { value: "conv-2" } });
    expect(onSelect).toHaveBeenCalledWith(mockConversations[1]);
  });

  it("handles new conversation creation form", async () => {
    const onCreate = vi.fn().mockResolvedValue(undefined);
    render(
      <ConversationNavigator
        conversations={mockConversations}
        activeConversation={mockConversations[0]}
        onSelectConversation={vi.fn()}
        onCreateConversation={onCreate}
        onRenameConversation={vi.fn()}
        onDeleteConversation={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "+ New Chat" }));
    const input = screen.getByPlaceholderText(/New chat title/i);
    fireEvent.change(input, { target: { value: "Deep Dive" } });
    fireEvent.click(screen.getByRole("button", { name: "Start" }));

    expect(onCreate).toHaveBeenCalledWith("Deep Dive");
  });

  it("handles renaming conversation", async () => {
    const onRename = vi.fn().mockResolvedValue(undefined);
    render(
      <ConversationNavigator
        conversations={mockConversations}
        activeConversation={mockConversations[0]}
        onSelectConversation={vi.fn()}
        onCreateConversation={vi.fn()}
        onRenameConversation={onRename}
        onDeleteConversation={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTitle("Rename conversation"));
    const input = screen.getByPlaceholderText("Chat title");
    fireEvent.change(input, { target: { value: "Renamed Title" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(onRename).toHaveBeenCalledWith("conv-1", "Renamed Title");
  });

  it("handles deleting conversation", async () => {
    const onDelete = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(
      <ConversationNavigator
        conversations={mockConversations}
        activeConversation={mockConversations[0]}
        onSelectConversation={vi.fn()}
        onCreateConversation={vi.fn()}
        onRenameConversation={vi.fn()}
        onDeleteConversation={onDelete}
      />,
    );

    fireEvent.click(screen.getByTitle("Delete conversation"));
    expect(onDelete).toHaveBeenCalledWith("conv-1");
  });
});
