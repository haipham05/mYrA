import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

function jsonResponse(payload: unknown) {
  return { ok: true, json: async () => payload } as Response;
}

function memorySourceFetch(projectIdForSource: string) {
  const sourcePaper = {
    id: "paper-outside-current-page",
    project_id: projectIdForSource,
    filename: "saved-source.pdf",
    status: "READY",
    document_sha256: "a".repeat(64),
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
  };
  const source = {
    id: "source-1",
    memory_id: "memory-1",
    source_type: "PAPER_CHUNK",
    paper_id: sourcePaper.id,
    page_number: 1,
    quote_text: "A saved quote without a verified anchor.",
    document_sha256: sourcePaper.document_sha256,
    anchor_status: "unresolved",
    bounding_boxes: [],
    anchors: [],
    created_at: "2026-10-01T00:00:00Z",
  };
  const memory = {
    id: "memory-1",
    project_id: "project-1",
    memory_type: "PAPER_FACT",
    status: "ACTIVE",
    title: "Saved paper fact",
    content: "A note with a saved paper source.",
    confidence: 1,
    importance: 0.8,
    version: 1,
    is_pinned: false,
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
    sources: [source],
    history: [],
  };

  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/health")) return jsonResponse({ status: "ok" });
    if (url.endsWith("/api/v1/system/status")) {
      return jsonResponse({ deepseek_configured: true });
    }
    if (url.endsWith("/api/v1/budget/usage")) {
      return jsonResponse({ status: "unavailable" });
    }
    if (url.endsWith("/api/v1/projects")) {
      return jsonResponse({
        items: [{ id: "project-1", name: "Research" }],
      });
    }
    if (url.endsWith("/api/v1/projects/project-1/papers")) {
      return jsonResponse({
        items: [
          {
            id: "paper-in-current-page",
            project_id: "project-1",
            filename: "visible-paper.pdf",
            status: "READY",
            document_sha256: "b".repeat(64),
            created_at: "2026-10-01T00:00:00Z",
            updated_at: "2026-10-01T00:00:00Z",
          },
        ],
        total: 2,
        offset: 0,
      });
    }
    if (url.includes("/api/v1/projects/project-1/conversations?")) {
      return jsonResponse({
        items: [
          {
            id: "conversation-1",
            project_id: "project-1",
            title: "Workspace Chat",
            paper_scope: "project",
            selected_paper_ids: [],
          },
        ],
      });
    }
    if (url.endsWith("/api/v1/conversations/conversation-1/messages")) {
      return jsonResponse([]);
    }
    if (url.includes("/api/v1/projects/project-1/memories?")) {
      return jsonResponse({ items: [memory], total: 1 });
    }
    if (url.endsWith(`/api/v1/papers/${sourcePaper.id}`)) {
      return jsonResponse(sourcePaper);
    }
    if (url.endsWith(`/api/v1/papers/${sourcePaper.id}/document`)) {
      return { ok: false, status: 503, json: async () => ({}) } as Response;
    }
    if (
      url.includes("/translations?project_id=") ||
      url.includes("translation-glossary")
    ) {
      return jsonResponse({ items: [], entries: [] });
    }
    return jsonResponse({});
  });
  return fetchMock;
}

describe("Home", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    window.localStorage.clear();
  });

  it("shows the connected state when the API responds", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((input: RequestInfo | URL) => {
        if (String(input).endsWith("/api/v1/budget/usage")) {
          return Promise.resolve(
            jsonResponse({
              status: "available",
              runtime_profile: "local",
              currency: "USD",
              daily_limit_estimate_usd: "1.00",
              daily_remaining_estimate_usd: "0.85",
              daily: {
                committed_estimate_usd: "0.15",
                active_reservation_usd: "0",
                unknown_reservation_usd: "0",
                reported_prompt_tokens: 100,
                reported_completion_tokens: 25,
              },
              usage_note: "Costs are estimates, not provider invoices.",
            }),
          );
        }
        return Promise.resolve(jsonResponse({ status: "ok" }));
      }),
    );

    render(<Home />);

    expect(screen.getByRole("heading", { name: "mYrA" })).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("Connected")).toBeInTheDocument(),
    );
    expect(screen.getByText("Provider usage")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Provider usage"));
    expect(screen.getByText("$0.8500 remaining of $1.00")).toBeInTheDocument();
    expect(screen.getByText("100")).toBeInTheDocument();
  });

  it("shows the unavailable state when the API request fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("connection failed")),
    );

    render(<Home />);

    await waitFor(() =>
      expect(screen.getByText("Unavailable")).toBeInTheDocument(),
    );
    expect(
      screen.queryByRole("heading", { name: "Translate this paper" }),
    ).not.toBeInTheDocument();
  });

  it("resolves a saved paper source outside the loaded library page and shows the exact-text fallback", async () => {
    const fetchMock = memorySourceFetch("project-1");
    vi.stubGlobal("fetch", fetchMock);

    render(<Home />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Notes and decisions" }),
    );
    fireEvent.click(
      await screen.findByRole("button", {
        name: "Open paper source (p. 1)",
      }),
    );

    expect(
      await screen.findByText("Exact highlight unavailable"),
    ).toBeInTheDocument();
    expect(screen.getByText("Cited on Page 1")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/papers/paper-outside-current-page",
      { cache: "no-store" },
    );
  });

  it("does not open a saved source paper from a different project", async () => {
    vi.stubGlobal("fetch", memorySourceFetch("another-project"));

    render(<Home />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Notes and decisions" }),
    );
    fireEvent.click(
      await screen.findByRole("button", {
        name: "Open paper source (p. 1)",
      }),
    );

    expect(
      await screen.findByText(
        "This source does not belong to the current project.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText("Cited on Page 1")).not.toBeInTheDocument();
  });

  it("keeps a saved run id when a temporary status check fails", async () => {
    const fetchBase = memorySourceFetch("project-1");
    const activeRun = {
      id: "run-1",
      project_id: "project-1",
      conversation_id: "conversation-1",
      status: "RUNNING",
      intent: "qa",
      action_summary: "Answer a paper question",
      stage: "assistant.tool.qa",
      result: null,
      safe_error: null,
      usage: null,
      created_at: "2026-10-05T00:00:00Z",
      updated_at: "2026-10-05T00:00:00Z",
    };
    let statusRequests = 0;
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      if (String(input).endsWith("/api/v1/runs/run-1")) {
        statusRequests += 1;
        return Promise.resolve(
          statusRequests === 1
            ? jsonResponse(activeRun)
            : ({ ok: false, status: 503, json: async () => ({}) } as Response),
        );
      }
      return fetchBase(input);
    });
    vi.stubGlobal("fetch", fetchMock);
    window.localStorage.setItem("myra.activeRun.conversation-1", "run-1");

    render(<Home />);

    expect(
      await screen.findByText(
        "Connection to the saved request was interrupted. Its status is kept; reload this conversation to check again.",
        {},
        { timeout: 3000 },
      ),
    ).toBeInTheDocument();
    expect(statusRequests).toBe(2);
    expect(window.localStorage.getItem("myra.activeRun.conversation-1")).toBe(
      "run-1",
    );
    expect(screen.getByText("Research in progress")).toBeInTheDocument();
  });

  it("shows translation controls for the selected project's selected paper", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/health")) return jsonResponse({ status: "ok" });
        if (url.endsWith("/api/v1/system/status")) {
          return jsonResponse({ deepseek_configured: true });
        }
        if (url.endsWith("/api/v1/projects")) {
          return jsonResponse({
            items: [{ id: "project-1", name: "Research" }],
          });
        }
        if (url.endsWith("/api/v1/projects/project-1/papers")) {
          return jsonResponse({
            items: [
              {
                id: "paper-1",
                project_id: "project-1",
                filename: "attention.pdf",
                status: "READY",
              },
            ],
          });
        }
        if (url.includes("/api/v1/projects/project-1/conversations?")) {
          return jsonResponse({ items: [] });
        }
        if (
          url.endsWith("/api/v1/projects/project-1/conversations") &&
          init?.method === "POST"
        ) {
          return jsonResponse({
            id: "conversation-1",
            project_id: "project-1",
            title: "Workspace Chat",
            paper_scope: "project",
            selected_paper_ids: [],
          });
        }
        if (url.endsWith("/api/v1/conversations/conversation-1/messages")) {
          return jsonResponse([]);
        }
        if (url.includes("/api/v1/papers/paper-1/translations")) {
          return jsonResponse([]);
        }
        if (url.endsWith("/api/v1/projects/project-1/translation-glossary")) {
          return jsonResponse({ entries: [] });
        }
        return jsonResponse({});
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<Home />);

    expect(
      await screen.findByRole("heading", { name: "Translate this paper" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Current paper: attention.pdf"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Collapse library" }));
    expect(screen.getByLabelText("Chat searches")).toHaveValue("project");
    expect(
      screen.getByText("Current paper: attention.pdf"),
    ).toBeInTheDocument();
    expect(
      screen.queryByPlaceholderText("Search title, author, or filename"),
    ).not.toBeInTheDocument();
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining(
          "http://127.0.0.1:8000/api/v1/papers/paper-1/translations",
        ),
        expect.anything(),
      ),
    );
    fireEvent.change(screen.getByLabelText("Chat searches"), {
      target: { value: "paper" },
    });
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/conversations/conversation-1?project_id=project-1",
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({
            paper_scope: "paper",
            selected_paper_ids: ["paper-1"],
          }),
        }),
      ),
    );
  });

  it("submits chat requests through the routed assistant run API", async () => {
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/health")) return jsonResponse({ status: "ok" });
        if (url.endsWith("/api/v1/system/status")) {
          return jsonResponse({ deepseek_configured: true });
        }
        if (url.endsWith("/api/v1/budget/usage")) {
          return jsonResponse({ status: "unavailable" });
        }
        if (url.endsWith("/api/v1/projects")) {
          return jsonResponse({
            items: [{ id: "project-1", name: "Research" }],
          });
        }
        if (url.endsWith("/api/v1/projects/project-1/papers")) {
          return jsonResponse({
            items: [
              {
                id: "paper-1",
                project_id: "project-1",
                filename: "attention.pdf",
                status: "READY",
              },
            ],
          });
        }
        if (url.includes("/api/v1/projects/project-1/conversations?")) {
          return jsonResponse({ items: [] });
        }
        if (
          url.endsWith("/api/v1/projects/project-1/conversations") &&
          init?.method === "POST"
        ) {
          return jsonResponse({
            id: "conversation-1",
            project_id: "project-1",
            title: "Workspace Chat",
            paper_scope: "project",
            selected_paper_ids: [],
          });
        }
        if (url.endsWith("/api/v1/conversations/conversation-1/messages")) {
          return jsonResponse([]);
        }
        if (url.endsWith("/api/v1/conversations/conversation-1/runs")) {
          return jsonResponse({
            id: "run-1",
            project_id: "project-1",
            conversation_id: "conversation-1",
            status: "SUCCEEDED",
            intent: "qa",
            action_summary: "Answer from the selected project",
            stage: null,
            result: {
              result_type: "answer",
              display_text: "Routed answer from the selected papers.",
              structured_payload: {},
              citations: [],
              warnings: [],
              usage: {},
              available_actions: [],
              artifact_ids: [],
            },
            safe_error: null,
            usage: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        }
        return jsonResponse({});
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<Home />);
    const input = await screen.findByPlaceholderText(
      "Ask a grounded research question…",
    );
    fireEvent.change(input, {
      target: { value: "What is the main contribution?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(
      await screen.findByText("Routed answer from the selected papers."),
    ).toBeInTheDocument();
    expect(screen.getByText("Using: Question answering")).toBeInTheDocument();
    const runCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/api/v1/conversations/conversation-1/runs"),
    );
    expect(runCall).toBeDefined();
    const requestBody = JSON.parse(String(runCall?.[1]?.body));
    expect(requestBody).toMatchObject({
      message: "What is the main contribution?",
      project_id: "project-1",
      conversation_id: "conversation-1",
      scope: "project",
      selected_paper_ids: [],
    });
    expect(requestBody.idempotency_key).toMatch(/^web-/);
  });

  it("restores a pending run and approves the exact stored action", async () => {
    window.localStorage.setItem("myra.activeRun.conversation-1", "run-saved");
    const run = {
      id: "run-saved",
      project_id: "project-1",
      conversation_id: "conversation-1",
      status: "AWAITING_APPROVAL",
      intent: "discover",
      action_summary: "Import the selected paper",
      stage: "assistant.approval",
      result: null,
      safe_error: null,
      usage: null,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    };
    const action = {
      id: "action-stored",
      run_id: "run-saved",
      action_type: "import_paper",
      arguments: {
        paper_id: "paper-1",
        source_url: "https://example.org/paper.pdf",
      },
      source_fingerprint: "a".repeat(64),
      status: "PENDING",
      expires_at: new Date(Date.now() + 60_000).toISOString(),
      decided_at: null,
    };
    let actionApproved = false;
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/health")) return jsonResponse({ status: "ok" });
        if (url.endsWith("/api/v1/system/status"))
          return jsonResponse({ deepseek_configured: true });
        if (url.endsWith("/api/v1/budget/usage"))
          return jsonResponse({ status: "unavailable" });
        if (url.endsWith("/api/v1/projects"))
          return jsonResponse({
            items: [{ id: "project-1", name: "Research" }],
          });
        if (url.endsWith("/api/v1/projects/project-1/papers")) {
          return jsonResponse({
            items: [
              {
                id: "paper-1",
                project_id: "project-1",
                filename: "paper.pdf",
                status: "READY",
              },
            ],
          });
        }
        if (url.includes("/api/v1/projects/project-1/conversations?")) {
          return jsonResponse({
            items: [
              {
                id: "conversation-1",
                project_id: "project-1",
                title: "Saved chat",
                paper_scope: "project",
                selected_paper_ids: [],
              },
            ],
          });
        }
        if (url.endsWith("/api/v1/conversations/conversation-1/messages"))
          return jsonResponse([]);
        if (url.endsWith("/api/v1/runs/run-saved/actions"))
          return jsonResponse([action]);
        if (url.endsWith("/api/v1/runs/run-saved") && init?.method !== "POST") {
          return jsonResponse({
            ...run,
            status: actionApproved ? "SUCCEEDED" : "AWAITING_APPROVAL",
            result: actionApproved
              ? {
                  result_type: "answer",
                  display_text: "Approved action completed.",
                  structured_payload: {},
                  citations: [],
                  warnings: [],
                  usage: {},
                  available_actions: [],
                  artifact_ids: [],
                }
              : null,
          });
        }
        if (url.endsWith("/api/v1/actions/action-stored/approve")) {
          actionApproved = true;
          return jsonResponse({ ...action, status: "APPROVED" });
        }
        return jsonResponse({});
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<Home />);

    expect(
      await screen.findByText("Review proposed action: import paper"),
    ).toBeInTheDocument();
    expect(screen.getByText(/source_url/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/actions/action-stored/approve",
        { method: "POST" },
      ),
    );
  });

  it("uses the viewed paper when switching from a multi-paper selection", async () => {
    const papers = ["paper-a", "paper-b", "paper-c"].map((id) => ({
      id,
      project_id: "project-1",
      filename: `${id}.pdf`,
      status: "READY",
    }));
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.endsWith("/health")) return jsonResponse({ status: "ok" });
        if (url.endsWith("/api/v1/system/status")) {
          return jsonResponse({ deepseek_configured: true });
        }
        if (url.endsWith("/api/v1/budget/usage")) {
          return jsonResponse({ status: "unavailable" });
        }
        if (url.endsWith("/api/v1/projects")) {
          return jsonResponse({
            items: [{ id: "project-1", name: "Research" }],
          });
        }
        if (url.endsWith("/api/v1/projects/project-1/papers")) {
          return jsonResponse({
            items: papers,
            total: papers.length,
            offset: 0,
          });
        }
        if (url.includes("/api/v1/projects/project-1/conversations?")) {
          return jsonResponse({
            items: [
              {
                id: "conversation-1",
                project_id: "project-1",
                title: "Workspace Chat",
                paper_scope: "selection",
                selected_paper_ids: ["paper-a", "paper-b"],
              },
            ],
          });
        }
        if (url.endsWith("/api/v1/conversations/conversation-1/messages")) {
          return jsonResponse([]);
        }
        if (
          url.endsWith(
            "/api/v1/conversations/conversation-1?project_id=project-1",
          ) &&
          init?.method === "PATCH"
        ) {
          const body = JSON.parse(String(init.body));
          return jsonResponse({
            id: "conversation-1",
            project_id: "project-1",
            title: "Workspace Chat",
            ...body,
          });
        }
        return jsonResponse({});
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<Home />);
    fireEvent.click(
      await screen.findByRole("button", { name: /paper-c\.pdf/ }),
    );
    fireEvent.change(screen.getByLabelText("Chat searches"), {
      target: { value: "paper" },
    });

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/conversations/conversation-1?project_id=project-1",
        expect.objectContaining({
          method: "PATCH",
          body: JSON.stringify({
            paper_scope: "paper",
            selected_paper_ids: ["paper-c"],
          }),
        }),
      ),
    );
  });
});
