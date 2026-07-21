import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PaperMetadataEditor from "./PaperMetadataEditor";
import type { Paper } from "@/types";

const paper: Paper = {
  id: "paper-1",
  project_id: "project-1",
  filename: "paper.pdf",
  status: "READY",
  title: "Extracted title",
  authors: null,
  metadata_provenance: { title: "docling_title" },
  created_at: "2026-10-05T00:00:00Z",
  updated_at: "2026-10-05T00:00:00Z",
};

function response(payload: unknown, ok = true) {
  return { ok, json: async () => payload } as Response;
}

describe("PaperMetadataEditor", () => {
  it("shows unknown metadata and saves owner corrections with project context", async () => {
    const updatedPaper = {
      ...paper,
      title: "Corrected title",
      authors: ["Ada Lovelace"],
      metadata_provenance: { title: "manual", authors: "manual" },
      updated_at: "2026-10-05T00:01:00Z",
    };
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(response(updatedPaper));
    const onUpdated = vi.fn();
    render(
      <PaperMetadataEditor
        paper={paper}
        apiUrl="http://127.0.0.1:8000"
        onUpdated={onUpdated}
      />,
    );

    fireEvent.click(screen.getByText("Paper details"));
    expect(screen.getByText("(docling_title)")).toBeInTheDocument();
    expect(screen.getAllByText("Not available").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Edit details" }));
    fireEvent.change(screen.getByLabelText("Title"), {
      target: { value: "Corrected title" },
    });
    fireEvent.change(screen.getByLabelText("Authors"), {
      target: { value: "Ada Lovelace" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save details" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/papers/paper-1?project_id=project-1",
        expect.objectContaining({
          method: "PATCH",
          body: expect.stringContaining('"authors":["Ada Lovelace"]'),
        }),
      ),
    );
    expect(onUpdated).toHaveBeenCalledWith(updatedPaper);
  });
});
