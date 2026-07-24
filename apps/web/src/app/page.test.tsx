import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

function jsonResponse(payload: unknown) {
  return { ok: true, json: async () => payload } as Response;
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
