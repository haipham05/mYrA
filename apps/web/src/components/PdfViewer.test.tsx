import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PdfViewer, { findRangeForQuote } from "./PdfViewer";
import type { Citation, Paper } from "@/types";

vi.mock("pdfjs-dist", () => {
  return {
    GlobalWorkerOptions: { workerSrc: "" },
    TextLayer: class MockTextLayer {
      container: HTMLElement;
      constructor({ container }: { container: HTMLElement }) {
        this.container = container;
      }
      async render() {
        const span1 = document.createElement("span");
        span1.textContent =
          "The dominant sequence transduction models are based on ";
        const span2 = document.createElement("span");
        span2.textContent =
          "complex recurrent or convolutional neural networks.";
        this.container.appendChild(span1);
        this.container.appendChild(span2);
      }
    },
    getDocument: () => ({
      promise: Promise.resolve({
        destroy: () => Promise.resolve(),
        getPage: () =>
          Promise.resolve({
            getViewport: () => ({ width: 612, height: 792 }),
            render: () => ({ promise: Promise.resolve() }),
            getTextContent: () =>
              Promise.resolve({
                items: [
                  {
                    str: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                  },
                ],
              }),
          }),
      }),
    }),
  };
});

const mockPaper: Paper = {
  id: "paper-123",
  project_id: "proj-123",
  filename: "attention_is_all_you_need.pdf",
  status: "READY",
  document_sha256: "aa".repeat(32),
  page_count: 5,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

const mockCitation: Citation = {
  citation_index: 1,
  evidence_id: "E1",
  paper_id: "paper-123",
  page_number: 2,
  quote:
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
  anchor_status: "verified",
  document_sha256: "aa".repeat(32),
  parser_version: "docling-2",
  anchors: [
    {
      id: "anchor-1",
      page_number: 2,
      source_element_id: "element-1",
      exact_quote:
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
      source_char_start: 0,
      source_char_end: 110,
      document_sha256: "aa".repeat(32),
      parser_version: "docling-2",
      anchor_status: "verified",
      bounding_boxes: [],
    },
  ],
  bounding_boxes: [
    {
      x_min: 72,
      y_min: 100,
      x_max: 300,
      y_max: 150,
      page_width: 612,
      page_height: 792,
      origin: "TOP_LEFT",
    },
  ],
};

describe("PdfViewer & Exact Range Matching", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer,
      }),
    );
    vi.stubGlobal("crypto", {
      subtle: {
        digest: async () => new Uint8Array(32).fill(0xaa).buffer,
      },
    });
    // Mock HTMLCanvasElement.getContext to eliminate jsdom warning
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
      fillRect: vi.fn(),
      clearRect: vi.fn(),
      getImageData: vi.fn(),
      putImageData: vi.fn(),
      createImageData: vi.fn(),
      setTransform: vi.fn(),
      drawImage: vi.fn(),
      save: vi.fn(),
      fillText: vi.fn(),
      restore: vi.fn(),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      closePath: vi.fn(),
      stroke: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      rotate: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
    });

    // Default DOMRect implementation for JSDOM
    Range.prototype.getClientRects = function () {
      return [
        {
          x: 72,
          y: 100,
          left: 72,
          top: 100,
          width: 228,
          height: 16,
          right: 300,
          bottom: 116,
          toJSON: () => {},
        } as DOMRect,
      ] as unknown as DOMRectList;
    };
    Range.prototype.getBoundingClientRect = function () {
      return {
        x: 72,
        y: 100,
        left: 72,
        top: 100,
        width: 228,
        height: 16,
        right: 300,
        bottom: 116,
        toJSON: () => {},
      } as DOMRect;
    };
  });

  afterEach(() => vi.unstubAllGlobals());

  describe("findRangeForQuote helper", () => {
    it("finds range spanning multiple text nodes with normalized whitespace", () => {
      const container = document.createElement("div");
      const span1 = document.createElement("span");
      span1.textContent = "Attention is\n";
      const span2 = document.createElement("span");
      span2.textContent = "all you need";
      container.appendChild(span1);
      container.appendChild(span2);

      const range = findRangeForQuote(container, "Attention is all you need");
      expect(range).not.toBeNull();
      expect(range?.toString()).toBe("Attention is\nall you need");
    });

    it("returns null when quote is not in container", () => {
      const container = document.createElement("div");
      container.textContent = "Some unrelated document text";

      const range = findRangeForQuote(container, "non-existent quote");
      expect(range).toBeNull();
    });

    it("refuses ambiguous repeated occurrences", () => {
      const container = document.createElement("div");
      const span1 = document.createElement("span");
      span1.textContent =
        "Introduction to the model architecture and its basic foundations. ";
      const span2 = document.createElement("span");
      span2.textContent =
        "Section 2: Introduction to the model architecture in deeper detail.";
      container.appendChild(span1);
      container.appendChild(span2);

      const targetQuote = "Introduction to the model architecture";

      expect(findRangeForQuote(container, targetQuote)).toBeNull();
    });
  });

  it("shows empty state when no paper is selected", () => {
    render(
      <PdfViewer
        paper={null}
        activeCitation={null}
        apiUrl="http://localhost:8000"
      />,
    );
    expect(screen.getByText("No paper selected")).toBeInTheDocument();
  });

  it("renders paper header and page controls", () => {
    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={null}
        apiUrl="http://localhost:8000"
      />,
    );
    expect(
      screen.getByText("attention_is_all_you_need.pdf"),
    ).toBeInTheDocument();
    expect(screen.getByText(/Page 1 of 5/)).toBeInTheDocument();
  });

  it("returns selected PDF text with its paper, page, and document identity", async () => {
    const onExplainSelection = vi.fn();
    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={null}
        apiUrl="http://localhost:8000"
        onExplainSelection={onExplainSelection}
      />,
    );

    const textLayer = await screen.findByTestId("pdf-text-layer");
    await waitFor(() =>
      expect(textLayer.textContent).toContain("dominant sequence"),
    );
    const textNode = textLayer.querySelector("span")?.firstChild;
    expect(textNode).toBeTruthy();
    const range = document.createRange();
    range.selectNodeContents(textNode!);
    window.getSelection()?.removeAllRanges();
    window.getSelection()?.addRange(range);

    fireEvent.mouseUp(textLayer);

    expect(onExplainSelection).toHaveBeenCalledWith({
      paper_id: "paper-123",
      page_number: 1,
      quote: "The dominant sequence transduction models are based on",
      document_sha256: "aa".repeat(32),
    });
  });

  it("renders citation callout and exact DOM Range highlights when active citation matches text layer", async () => {
    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={mockCitation}
        apiUrl="http://localhost:8000"
      />,
    );
    // Page in toolbar is strictly synchronized with citation page
    expect(screen.getByText(/Page 2 of 5/)).toBeInTheDocument();
    expect(screen.getByText(/Cited on Page 2/)).toBeInTheDocument();
    expect(
      screen.getByText(/The dominant sequence transduction/),
    ).toBeInTheDocument();

    // Exact highlight rendered from DOM Range.getClientRects()
    await waitFor(() => {
      expect(screen.getByTestId("evidence-highlight")).toBeInTheDocument();
      expect(screen.getByText("Verbatim match")).toBeInTheDocument();
    });
  });

  it("renders multiple highlight rects for wrapped text spanning multiple lines", async () => {
    Range.prototype.getClientRects = function () {
      return [
        {
          x: 72,
          y: 100,
          left: 72,
          top: 100,
          width: 300,
          height: 16,
          right: 372,
          bottom: 116,
          toJSON: () => {},
        } as DOMRect,
        {
          x: 72,
          y: 118,
          left: 72,
          top: 118,
          width: 150,
          height: 16,
          right: 222,
          bottom: 134,
          toJSON: () => {},
        } as DOMRect,
      ] as unknown as DOMRectList;
    };

    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={mockCitation}
        apiUrl="http://localhost:8000"
      />,
    );

    await waitFor(() => {
      const highlights = screen.getAllByTestId("evidence-highlight");
      expect(highlights).toHaveLength(2);
    });
  });

  it("shows explicit 'Exact highlight unavailable' badge and avoids drawing false boxes when unresolved", async () => {
    const unresolvedCitation: Citation = {
      ...mockCitation,
      citation_index: 2,
      evidence_id: "E2",
      page_number: 4,
      anchor_status: "unresolved",
      bounding_boxes: [],
      quote: "Scanned text with no reliable coordinates.",
    };

    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={unresolvedCitation}
        apiUrl="http://localhost:8000"
      />,
    );

    // Toolbar must synchronize to cited page 4
    expect(screen.getByText(/Page 4 of 5/)).toBeInTheDocument();
    expect(screen.getByText(/Cited on Page 4/)).toBeInTheDocument();
    // Shows explicit unavailable state
    expect(screen.getByText("Exact highlight unavailable")).toBeInTheDocument();
    // NEVER draw false highlight boxes
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });

  it("does not render highlight when citation belongs to a different paper", () => {
    const citationFromOtherPaper: Citation = {
      ...mockCitation,
      paper_id: "other-paper-999",
    };

    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={citationFromOtherPaper}
        apiUrl="http://localhost:8000"
      />,
    );

    // Should stay on page 1 and not show highlights or citation callout
    expect(screen.getByText(/Page 1 of 5/)).toBeInTheDocument();
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
    expect(screen.queryByText(/Cited on Page/)).not.toBeInTheDocument();
  });

  it("does not render page 2 highlight when user navigates away to another page", async () => {
    const { getByRole } = render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={mockCitation}
        apiUrl="http://localhost:8000"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/Page 2 of 5/)).toBeInTheDocument();
      expect(screen.getByTestId("evidence-highlight")).toBeInTheDocument();
    });

    // Click Next button to navigate to page 3
    const nextButton = getByRole("button", { name: "Next" });
    fireEvent.click(nextButton);

    expect(screen.getByText(/Page 3 of 5/)).toBeInTheDocument();
    // Highlight must NOT appear on page 3
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });

  it("rejects highlight when paper SHA-256 and citation SHA-256 mismatch", () => {
    const paperWithHash: Paper = {
      ...mockPaper,
      document_sha256: "hash_version_A",
    };
    const citationWithDifferentHash: Citation = {
      ...mockCitation,
      document_sha256: "hash_version_B",
    };

    render(
      <PdfViewer
        paper={paperWithHash}
        activeCitation={citationWithDifferentHash}
        apiUrl="http://localhost:8000"
      />,
    );

    // Shows explicit unavailable state and does not draw highlight
    expect(screen.getByText("Exact highlight unavailable")).toBeInTheDocument();
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });

  it("rejects a changed served PDF even when metadata hashes agree", async () => {
    vi.stubGlobal("crypto", {
      subtle: {
        digest: async () => new Uint8Array(32).fill(0xbb).buffer,
      },
    });
    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={mockCitation}
        apiUrl="http://localhost:8000"
      />,
    );
    await waitFor(() => expect(fetch).toHaveBeenCalled());
    expect(screen.getByText("Exact highlight unavailable")).toBeInTheDocument();
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });

  it("rejects a verified badge without a matching verified source anchor", () => {
    const citationWithWrongAnchor: Citation = {
      ...mockCitation,
      anchors: [
        { ...mockCitation.anchors![0], exact_quote: "Different evidence." },
      ],
    };
    render(
      <PdfViewer
        paper={mockPaper}
        activeCitation={citationWithWrongAnchor}
        apiUrl="http://localhost:8000"
      />,
    );
    expect(screen.getByText("Exact highlight unavailable")).toBeInTheDocument();
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });
});
