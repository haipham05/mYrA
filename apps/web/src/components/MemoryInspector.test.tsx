import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import MemoryInspector from "./MemoryInspector";
import type { Memory } from "@/types";

const mockMemories: Memory[] = [
  {
    id: "mem-1",
    project_id: "proj-1",
    memory_type: "DECISION",
    status: "ACTIVE",
    title: "Decision: Choose AURC over ECE",
    content:
      "Project decision: Selected AURC over ECE for calibration evaluation.",
    confidence: 0.95,
    importance: 0.9,
    version: 1,
    is_pinned: true,
    created_at: "2026-09-27T10:00:00Z",
    updated_at: "2026-09-27T10:00:00Z",
    sources: [
      {
        id: "src-1",
        memory_id: "mem-1",
        source_type: "MESSAGE",
        message_id: "msg-1",
        created_at: "2026-09-27T10:00:00Z",
      },
    ],
    history: [
      {
        id: "aud-1",
        memory_id: "mem-1",
        action: "CREATED",
        new_content:
          "Project decision: Selected AURC over ECE for calibration evaluation.",
        reason: "Initial decision",
        created_at: "2026-09-27T10:00:00Z",
      },
    ],
  },
  {
    id: "mem-2",
    project_id: "proj-1",
    memory_type: "PREFERENCE",
    status: "SUPERSEDED",
    title: "Preference: Concise Bullets",
    content: "User preference: Provide concise bullet points.",
    confidence: 0.85,
    importance: 0.7,
    version: 2,
    is_pinned: false,
    superseded_by_id: "mem-3",
    created_at: "2026-09-26T10:00:00Z",
    updated_at: "2026-09-27T09:00:00Z",
    sources: [],
    history: [],
  },
];

describe("MemoryInspector Component", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders empty state when no memories are returned", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: [], total: 0 }),
    } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    expect(
      screen.getByRole("heading", { name: "Notes and decisions" }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(
        screen.getByText(/No memories found matching the selected filters/i),
      ).toBeInTheDocument();
    });
  });

  it("renders memory cards with badges, version, and content", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: mockMemories, total: 2 }),
    } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
      expect(
        screen.getByText("Preference: Concise Bullets"),
      ).toBeInTheDocument();
    });

    expect(
      screen.getByText("Decision", { selector: "span" }),
    ).toBeInTheDocument();
    expect(screen.getByText("ACTIVE")).toBeInTheDocument();
    expect(screen.getByText("v1")).toBeInTheDocument();
    expect(screen.getByText("SUPERSEDED")).toBeInTheDocument();
    expect(screen.getByText("v2")).toBeInTheDocument();
  });

  it("toggles pin status via PATCH", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...mockMemories[0],
          is_pinned: false,
          version: 2,
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const pinBtn = screen.getByTitle("Unpin memory");
    fireEvent.click(pinBtn);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/projects/proj-1/memories/mem-1",
        expect.objectContaining({
          method: "PATCH",
          body: expect.stringContaining('"is_pinned":false'),
        }),
      );
    });
  });

  it("displays stale edit warning on 409 conflict during edit", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response)
      .mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({
          detail: "Version conflict: memory has been modified",
        }),
      } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    // Click Edit button for active memory
    const editBtns = screen.getAllByRole("button", { name: "Edit" });
    fireEvent.click(editBtns[0]);

    expect(screen.getByText(/Edit note \(v1\)/i)).toBeInTheDocument();

    // Submit edit
    const saveBtn = screen.getByRole("button", { name: "Save Changes" });
    fireEvent.click(saveBtn);

    await waitFor(() => {
      expect(
        screen.getByText(
          /Stale edit warning: this memory was modified concurrently/i,
        ),
      ).toBeInTheDocument();
    });
  });

  it("expands and collapses audit history", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: mockMemories, total: 2 }),
    } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const historyBtn = screen.getByRole("button", {
      name: /Show Audit History/i,
    });
    fireEvent.click(historyBtn);

    expect(screen.getByText("CREATED")).toBeInTheDocument();
    expect(screen.getByText(/Reason: Initial decision/i)).toBeInTheDocument();

    // Collapse
    const hideBtn = screen.getByRole("button", { name: /Hide Audit History/i });
    fireEvent.click(hideBtn);
    expect(
      screen.queryByText(/Reason: Initial decision/i),
    ).not.toBeInTheDocument();
  });

  it("invokes onSelectSource callback when clicking provenance jump button", async () => {
    const onSelectSource = vi.fn();
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: mockMemories, total: 2 }),
    } as Response);

    render(
      <MemoryInspector
        projectId="proj-1"
        apiUrl="http://127.0.0.1:8000"
        onSelectSource={onSelectSource}
      />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const jumpBtn = screen.getByRole("button", { name: /Open conversation/i });
    fireEvent.click(jumpBtn);
    expect(onSelectSource).toHaveBeenCalledWith(mockMemories[0].sources[0]);
  });

  it("shows a researcher-facing paper source link with its page", async () => {
    const onSelectSource = vi.fn();
    const paperFact: Memory = {
      ...mockMemories[0],
      id: "paper-fact-1",
      memory_type: "PAPER_FACT",
      sources: [
        {
          id: "paper-source-1",
          memory_id: "paper-fact-1",
          source_type: "PAPER_CHUNK",
          paper_id: "paper-1",
          page_number: 7,
          quote_text: "A source-backed statement.",
          created_at: "2026-09-27T10:00:00Z",
        },
      ],
    };
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: [paperFact], total: 1 }),
    } as Response);

    render(
      <MemoryInspector
        projectId="proj-1"
        apiUrl="http://127.0.0.1:8000"
        onSelectSource={onSelectSource}
      />,
    );

    const openSource = await screen.findByRole("button", {
      name: "Open paper source (p. 7)",
    });
    expect(screen.getByText("Paper source:")).toBeInTheDocument();
    expect(screen.getByText("Paper fact")).toBeInTheDocument();
    fireEvent.click(openSource);
    expect(onSelectSource).toHaveBeenCalledWith(paperFact.sources[0]);
  });

  it("sends expected_version in supersede URL and refreshes list", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          ...mockMemories[0],
          id: "mem-3",
          version: 1,
          status: "ACTIVE",
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const supersedeBtns = screen.getAllByRole("button", { name: "Supersede" });
    fireEvent.click(supersedeBtns[0]);

    expect(screen.getByText(/Supersede Decision \(v1\)/i)).toBeInTheDocument();

    const contentInput = screen.getByRole("textbox", {
      name: "New Content / Rationale",
    });
    fireEvent.change(contentInput, {
      target: { value: "Switching to Brier score for better calibration." },
    });

    const submitBtn = screen.getByRole("button", {
      name: "Supersede Decision",
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/projects/proj-1/memories/mem-1/supersede?expected_version=1",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" },
        }),
      );
    });
  });

  it("displays client validation error when attempting to supersede with identical content", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({ items: mockMemories, total: 2 }),
    } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const supersedeBtns = screen.getAllByRole("button", { name: "Supersede" });
    fireEvent.click(supersedeBtns[0]);

    const submitBtn = screen.getByRole("button", {
      name: "Supersede Decision",
    });
    // Submit without modifying content
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(/Content must be modified to supersede this memory/i),
      ).toBeInTheDocument();
    });
  });

  it("displays stale version conflict error on 409 conflict during supersede", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: mockMemories, total: 2 }),
      } as Response)
      .mockResolvedValueOnce({
        ok: false,
        status: 409,
        json: async () => ({
          detail: "MemoryVersionConflictError: Stale version",
        }),
      } as Response);

    render(
      <MemoryInspector projectId="proj-1" apiUrl="http://127.0.0.1:8000" />,
    );

    await waitFor(() => {
      expect(
        screen.getByText("Decision: Choose AURC over ECE"),
      ).toBeInTheDocument();
    });

    const supersedeBtns = screen.getAllByRole("button", { name: "Supersede" });
    fireEvent.click(supersedeBtns[0]);

    const contentInput = screen.getByRole("textbox", {
      name: "New Content / Rationale",
    });
    fireEvent.change(contentInput, {
      target: { value: "Switching to Brier score for better calibration." },
    });

    const submitBtn = screen.getByRole("button", {
      name: "Supersede Decision",
    });
    fireEvent.click(submitBtn);

    await waitFor(() => {
      expect(
        screen.getByText(
          /Stale version conflict: this memory was modified concurrently/i,
        ),
      ).toBeInTheDocument();
    });
  });
});
