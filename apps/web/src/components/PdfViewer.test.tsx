import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PdfViewer from "./PdfViewer";
import type { Citation, Paper } from "@/types";

const mockPaper: Paper = {
  id: "paper-123",
  project_id: "proj-123",
  filename: "attention_is_all_you_need.pdf",
  status: "READY",
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

describe("PdfViewer", () => {
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

  it("renders citation callout and bounding box overlay when active citation matches paper", () => {
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
    // Bounding box overlay rendered on page 2
    expect(screen.getByTestId("evidence-highlight")).toBeInTheDocument();
  });

  it("shows explicit 'Exact highlight unavailable' badge and avoids drawing false boxes when unresolved", () => {
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

    // Initially on page 2 with highlight
    expect(screen.getByText(/Page 2 of 5/)).toBeInTheDocument();
    expect(screen.getByTestId("evidence-highlight")).toBeInTheDocument();

    // Click Next button to navigate to page 3
    const nextButton = getByRole("button", { name: "Next" });
    fireEvent.click(nextButton);

    expect(screen.getByText(/Page 3 of 5/)).toBeInTheDocument();
    // Highlight must NOT appear on page 3
    expect(screen.queryByTestId("evidence-highlight")).not.toBeInTheDocument();
  });
});
