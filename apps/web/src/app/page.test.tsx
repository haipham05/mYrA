import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

function jsonResponse(payload: unknown) {
  return { ok: true, json: async () => payload } as Response;
}

describe("Home", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows the connected state when the API responds", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue({ ok: true, json: async () => ({ status: "ok" }) }),
    );

    render(<Home />);

    expect(screen.getByRole("heading", { name: "mYrA" })).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText("Connected")).toBeInTheDocument(),
    );
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
            title: "Workspace Chat",
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
  });
});
