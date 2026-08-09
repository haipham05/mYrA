import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import TranslationPanel from "./TranslationPanel";

const props = {
  apiUrl: "http://127.0.0.1:8000",
  projectId: "project-1",
  paperId: "paper-1",
  paperStatus: "READY",
};

function response(payload: unknown, ok = true) {
  return { ok, json: async () => payload } as Response;
}

describe("TranslationPanel", () => {
  beforeEach(() => vi.restoreAllMocks());

  it("requires external-processing acknowledgement before submitting English to Vietnamese", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(response({ entries: [] }))
      .mockResolvedValueOnce(
        response({ id: "translation-1", status: "PENDING" }),
      );

    render(<TranslationPanel {...props} />);
    expect(await screen.findByText(/No translations yet/i)).toBeInTheDocument();
    expect(
      screen.getByText("English → Vietnamese · translated-only PDF"),
    ).toBeInTheDocument();

    const submit = screen.getByRole("button", {
      name: "Translate to Vietnamese",
    });
    expect(submit).toBeDisabled();
    fireEvent.click(
      screen.getByRole("checkbox", { name: /external translation service/i }),
    );
    expect(submit).toBeEnabled();
    fireEvent.click(submit);

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/papers/paper-1/translations?project_id=project-1",
        expect.objectContaining({
          method: "POST",
          body: expect.stringContaining(
            '"acknowledge_external_processing":true',
          ),
        }),
      ),
    );
    expect(await screen.findByText(/Status: pending/i)).toBeInTheDocument();
  });

  it("loads and saves editable glossary entries", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(
        response({ entries: [{ english: "attention", vietnamese: "chú ý" }] }),
      )
      .mockResolvedValueOnce(
        response({
          entries: [{ english: "attention", vietnamese: "sự chú ý" }],
        }),
      );

    render(<TranslationPanel {...props} />);
    const vietnameseInput = await screen.findByLabelText(
      "Preferred Vietnamese",
    );
    fireEvent.change(vietnameseInput, { target: { value: "sự chú ý" } });
    fireEvent.click(screen.getByRole("button", { name: "Save glossary" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/projects/project-1/translation-glossary",
        expect.objectContaining({
          method: "PUT",
          body: expect.stringContaining("sự chú ý"),
        }),
      ),
    );
    expect(await screen.findByDisplayValue("sự chú ý")).toBeInTheDocument();
  });

  it("shows active status and offers cancellation, then a completed PDF download", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(
        response([
          {
            id: "translation-1",
            status: "PROCESSING",
            stage: "LAYOUT_ANALYSIS",
            completed_units: 2,
            total_units: 8,
          },
        ]),
      )
      .mockResolvedValueOnce(response({ entries: [] }))
      .mockResolvedValueOnce(
        response({ id: "translation-1", status: "CANCELLED" }),
      );

    render(<TranslationPanel {...props} />);
    expect(
      await screen.findByText(/Status: processing · layout analysis/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Progress: 2 of 8 sections")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:8000/api/v1/translations/translation-1/cancel?project_id=project-1",
        expect.objectContaining({ method: "POST" }),
      ),
    );

    const completedFetch = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(
        response([
          {
            id: "translation-2",
            status: "COMPLETED",
            completed_units: 2,
            total_units: 8,
            skipped_units: 6,
            warnings: [
              "6 of 8 detected text sections were preserved or skipped rather than translated. Review the PDF for completeness.",
            ],
            segments: [
              {
                ordinal: 0,
                source_page_number: 2,
                source_quote: "Attention is all you need.",
                status: "VALIDATED",
              },
            ],
          },
        ]),
      )
      .mockResolvedValueOnce(response({ entries: [] }))
      .mockResolvedValueOnce(
        response({
          segments: [
            {
              ordinal: 0,
              source_page_number: 2,
              source_quote: "Attention is all you need.",
              anchor_status: "verified",
              source_char_start: 123,
              source_char_end: 149,
              document_sha256: "a".repeat(64),
              parser_version: "translation-page-rawtext-v1",
            },
          ],
        }),
      );
    const onOpenSource = vi.fn();
    const { rerender } = render(
      <TranslationPanel {...props} onOpenSource={onOpenSource} />,
    );
    expect(
      await screen.findByRole("link", { name: "Download PDF" }),
    ).toHaveAttribute(
      "href",
      "http://127.0.0.1:8000/api/v1/translations/translation-2/pdf?project_id=project-1",
    );
    expect(
      screen.getByText(
        "6 of 8 detected text sections were preserved or skipped rather than translated. Review the PDF for completeness.",
      ),
    ).toBeInTheDocument();
    expect(completedFetch).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole("button", { name: "Preview PDF" }));
    expect(
      await screen.findByTitle("Translated Vietnamese PDF preview"),
    ).toHaveAttribute(
      "src",
      "http://127.0.0.1:8000/api/v1/translations/translation-2/pdf?project_id=project-1&inline=true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Open original page" }));
    await waitFor(() =>
      expect(onOpenSource).toHaveBeenCalledWith(
        expect.objectContaining({
          source_page_number: 2,
          source_quote: "Attention is all you need.",
          anchor_status: "verified",
          source_char_start: 123,
        }),
      ),
    );
    rerender(<TranslationPanel {...props} paperStatus="PROCESSING" />);
  });

  it("offers retry for failed jobs and reports API errors accessibly", async () => {
    const fetchMock = vi
      .spyOn(global, "fetch")
      .mockResolvedValueOnce(
        response([
          {
            id: "translation-3",
            status: "FAILED",
            error_message: "Provider temporarily unavailable",
          },
        ]),
      )
      .mockResolvedValueOnce(response({ entries: [] }))
      .mockResolvedValueOnce(
        response({ detail: "Retry is not available yet" }, false),
      );

    render(<TranslationPanel {...props} />);
    expect(
      await screen.findByText("Provider temporarily unavailable"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Retry is not available yet",
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/v1/translations/translation-3/retry?project_id=project-1",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("disables translation until a paper is ready", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(response([]))
      .mockResolvedValueOnce(response({ entries: [] }));
    render(<TranslationPanel {...props} paperStatus="PROCESSING" />);
    expect(
      await screen.findByText(/must finish indexing/i),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Translate to Vietnamese" }),
    ).toBeDisabled();
  });
});
