import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PaperUploader from "./PaperUploader";
import type { Paper } from "@/types";

const paper: Paper = {
  id: "paper-1",
  project_id: "project-1",
  filename: "legacy.pdf",
  status: "READY",
  title: "A Useful Paper",
  created_at: "2026-10-05T00:00:00Z",
  updated_at: "2026-10-05T00:00:00Z",
};

describe("PaperUploader library controls", () => {
  it("searches metadata and filename with bounded pagination", async () => {
    const onSearch = vi.fn().mockResolvedValue(undefined);
    const onScopeChange = vi.fn().mockResolvedValue(undefined);
    const onSelectedPaperIdsChange = vi.fn().mockResolvedValue(undefined);
    render(
      <PaperUploader
        projectId="project-1"
        apiUrl="http://127.0.0.1:8000"
        papers={[paper]}
        total={60}
        offset={0}
        selectedPaper={paper}
        onPaperSelect={vi.fn()}
        onUploadSuccess={vi.fn()}
        onSearch={onSearch}
        paperScope="selection"
        selectedPaperIds={[]}
        onScopeChange={onScopeChange}
        onSelectedPaperIdsChange={onSelectedPaperIdsChange}
      />,
    );

    expect(screen.getByText("A Useful Paper")).toBeInTheDocument();
    fireEvent.change(
      screen.getByPlaceholderText("Search title, author, or filename"),
      {
        target: { value: "retrieval" },
      },
    );
    fireEvent.change(screen.getByLabelText("Filter by status"), {
      target: { value: "READY" },
    });
    fireEvent.change(screen.getByLabelText("Filter by year"), {
      target: { value: "2024" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search" }));
    await waitFor(() =>
      expect(onSearch).toHaveBeenCalledWith({
        q: "retrieval",
        status: "READY",
        year: "2024",
        offset: 0,
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await waitFor(() =>
      expect(onSearch).toHaveBeenLastCalledWith({
        q: "retrieval",
        status: "READY",
        year: "2024",
        offset: 50,
      }),
    );

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Include A Useful Paper in chat scope",
      }),
    );
    expect(onSelectedPaperIdsChange).toHaveBeenCalledWith("paper-1", true);
    fireEvent.change(screen.getByLabelText("Chat searches"), {
      target: { value: "paper" },
    });
    expect(onScopeChange).toHaveBeenCalledWith("paper");
  });

  it("shows recovered processing work rather than timing it out", async () => {
    const onUploadSuccess = vi.fn().mockResolvedValue(undefined);
    const processingPaper: Paper = {
      ...paper,
      status: "PROCESSING",
      latest_job: {
        id: "job-1",
        status: "PROCESSING",
        stage: "EMBEDDING",
        progress: 0.45,
        is_retryable: false,
        retry_count: 0,
      },
    };
    render(
      <PaperUploader
        projectId="project-1"
        apiUrl="http://127.0.0.1:8000"
        papers={[processingPaper]}
        total={1}
        offset={0}
        selectedPaper={processingPaper}
        onPaperSelect={vi.fn()}
        onUploadSuccess={onUploadSuccess}
        onSearch={vi.fn().mockResolvedValue(undefined)}
        paperScope="project"
        selectedPaperIds={[]}
        onScopeChange={vi.fn().mockResolvedValue(undefined)}
        onSelectedPaperIdsChange={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    expect(
      await screen.findByText(/1 ingestion job still processing · EMBEDDING/),
    ).toBeInTheDocument();
    expect(screen.getByText("45%")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Upload PDF" })).toBeEnabled();
  });

  it("offers retry only for a retryable failed job", async () => {
    const onUploadSuccess = vi.fn().mockResolvedValue(undefined);
    const failedPaper: Paper = {
      ...paper,
      status: "FAILED",
      latest_job: {
        id: "job-failed",
        status: "FAILED",
        stage: "FAILED",
        progress: 0,
        error_message: "temporary provider error",
        is_retryable: true,
        retry_count: 1,
      },
    };
    const fetchMock = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    const { rerender } = render(
      <PaperUploader
        projectId="project-1"
        apiUrl="http://127.0.0.1:8000"
        papers={[failedPaper]}
        total={1}
        offset={0}
        selectedPaper={failedPaper}
        onPaperSelect={vi.fn()}
        onUploadSuccess={onUploadSuccess}
        onSearch={vi.fn().mockResolvedValue(undefined)}
        paperScope="project"
        selectedPaperIds={[]}
        onScopeChange={vi.fn().mockResolvedValue(undefined)}
        onSelectedPaperIdsChange={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/jobs/job-failed/retry",
        { method: "POST" },
      ),
    );
    expect(onUploadSuccess).toHaveBeenCalledOnce();

    rerender(
      <PaperUploader
        projectId="project-1"
        apiUrl="http://127.0.0.1:8000"
        papers={[
          {
            ...failedPaper,
            latest_job: { ...failedPaper.latest_job!, is_retryable: false },
          },
        ]}
        total={1}
        offset={0}
        selectedPaper={failedPaper}
        onPaperSelect={vi.fn()}
        onUploadSuccess={onUploadSuccess}
        onSearch={vi.fn().mockResolvedValue(undefined)}
        paperScope="project"
        selectedPaperIds={[]}
        onScopeChange={vi.fn().mockResolvedValue(undefined)}
        onSelectedPaperIdsChange={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    expect(
      screen.queryByRole("button", { name: "Retry" }),
    ).not.toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});
