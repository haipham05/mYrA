import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GraphExplorer from "./GraphExplorer";
import {
  fetchFactDetail,
  fetchGraphStatus,
  fetchNodeNeighbors,
  searchGraphNodes,
  triggerGraphIndex,
  GraphUnavailableError,
} from "@/lib/graph";
import type {
  GraphFactDetail,
  GraphNeighbor,
  GraphNode,
  GraphStatus,
} from "@/types";

// Mock @/lib/graph methods
vi.mock("@/lib/graph", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/graph")>();
  return {
    ...actual,
    fetchGraphStatus: vi.fn(),
    searchGraphNodes: vi.fn(),
    fetchNodeNeighbors: vi.fn(),
    fetchFactDetail: vi.fn(),
    triggerGraphIndex: vi.fn(),
  };
});

// Mock PdfViewer component
vi.mock("@/components/PdfViewer", () => {
  return {
    default: ({
      paper,
      activeCitation,
    }: {
      paper: { filename: string } | null;
      activeCitation: { quote: string } | null;
    }) => (
      <div data-testid="mock-pdf-viewer">
        <div data-testid="mock-pdf-filename">{paper?.filename}</div>
        <div data-testid="mock-pdf-citation">{activeCitation?.quote}</div>
      </div>
    ),
  };
});

const mockApiUrl = "http://127.0.0.1:8000";
const mockProjectId = "proj-123";

const mockStatusConnected: GraphStatus = {
  project_id: mockProjectId,
  graphrag_enabled: true,
  neo4j_available: true,
  node_count: 42,
  fact_count: 128,
  pending_events_count: 0,
  completed_events_count: 10,
  failed_events_count: 0,
};

const mockStatusOffline: GraphStatus = {
  project_id: mockProjectId,
  graphrag_enabled: true,
  neo4j_available: false,
  node_count: 15,
  fact_count: 30,
  pending_events_count: 2,
  completed_events_count: 5,
  failed_events_count: 1,
};

const mockNodes: GraphNode[] = [
  {
    key: "node-transformer",
    project_id: mockProjectId,
    name: "Transformer",
    type: "Model",
    description: "Novel sequence transduction architecture based on attention",
    aliases: ["Vaswani Transformer", "Vanilla Transformer"],
    updated_at: "2026-09-29T10:00:00Z",
  },
  {
    key: "node-bleu",
    project_id: mockProjectId,
    name: "BLEU",
    type: "Metric",
    description: "Bilingual evaluation understudy score",
    aliases: [],
    updated_at: "2026-09-29T10:00:00Z",
  },
  {
    key: "node-wmt14",
    project_id: mockProjectId,
    name: "WMT 2014 English-to-German",
    type: "Dataset",
    description: "Standard translation benchmark",
    aliases: ["WMT14 En-De"],
    updated_at: "2026-09-29T10:00:00Z",
  },
];

const mockNeighbors: GraphNeighbor[] = [
  {
    neighbor_key: "node-wmt14",
    neighbor_name: "WMT 2014 English-to-German",
    neighbor_type: "Dataset",
    direction: "OUTGOING",
    predicate: "EVALUATED_ON",
    fact_id: "fact-1",
    qualifiers: { split: "test" },
  },
  {
    neighbor_key: "node-attention-paper",
    neighbor_name: "Attention Is All You Need",
    neighbor_type: "Paper",
    direction: "INCOMING",
    predicate: "PROPOSES_METHOD",
    fact_id: "fact-2",
  },
];

const mockVerifiedFact: GraphFactDetail = {
  id: "fact-1",
  project_id: mockProjectId,
  paper_id: "paper-123",
  paper_title: "Attention Is All You Need",
  generation_id: "gen-1",
  predicate: "EVALUATED_ON",
  subject_key: "node-transformer",
  subject_name: "Transformer",
  subject_type: "Model",
  object_key: "node-wmt14",
  object_name: "WMT 2014 English-to-German",
  object_type: "Dataset",
  qualifiers: {
    metric: "BLEU",
    result_value: 28.4,
  },
  exact_quote:
    "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
  page_number: 6,
  char_start: 120,
  char_end: 220,
  document_sha256: "aabbcc112233".padEnd(64, "0"),
  citation: {
    citation_index: 1,
    evidence_id: "ev-1",
    paper_id: "paper-123",
    page_number: 6,
    bounding_boxes: [],
    quote:
      "On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU.",
    anchor_status: "verified",
  },
  anchor_status: "verified",
};

const mockUnverifiedFact: GraphFactDetail = {
  id: "fact-2",
  project_id: mockProjectId,
  paper_id: "paper-123",
  paper_title: "Attention Is All You Need",
  generation_id: "gen-1",
  predicate: "PROPOSES_METHOD",
  subject_key: "node-attention-paper",
  subject_name: "Attention Is All You Need",
  subject_type: "Paper",
  object_key: "node-transformer",
  object_name: "Transformer",
  object_type: "Model",
  qualifiers: null,
  exact_quote:
    "We propose the Transformer, a model architecture eschewing recurrence.",
  page_number: 1,
  char_start: 50,
  char_end: 115,
  document_sha256: "aabbcc112233".padEnd(64, "0"),
  citation: null,
  anchor_status: "unresolved",
};

describe("GraphExplorer Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();

    vi.mocked(fetchGraphStatus).mockResolvedValue(mockStatusConnected);
    vi.mocked(searchGraphNodes).mockResolvedValue({
      items: mockNodes,
      total: mockNodes.length,
      limit: 10,
      skip: 0,
    });
    vi.mocked(fetchNodeNeighbors).mockResolvedValue({
      node_key: "node-transformer",
      neighbors: mockNeighbors,
      total: mockNeighbors.length,
    });
    vi.mocked(fetchFactDetail).mockResolvedValue(mockVerifiedFact);
  });

  // Test 1: Renders loading and empty states when no nodes match.
  it("renders loading and empty states when no nodes match", async () => {
    // Setup empty search result
    vi.mocked(searchGraphNodes).mockResolvedValueOnce({
      items: [],
      total: 0,
      limit: 10,
      skip: 0,
    });

    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    // Initial loading or immediate empty state
    await waitFor(() => {
      expect(
        screen.getByText("No entities found matching the selected filters."),
      ).toBeInTheDocument();
    });
  });

  // Test 2: Renders node list with type badges and pagination.
  it("renders node list with type badges and pagination", async () => {
    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    // Switch to List View
    const listViewBtn = screen.getByRole("button", { name: "List View" });
    fireEvent.click(listViewBtn);

    await waitFor(() => {
      expect(screen.getByText("Transformer")).toBeInTheDocument();
      expect(screen.getByText("BLEU")).toBeInTheDocument();
      expect(
        screen.getByText("WMT 2014 English-to-German"),
      ).toBeInTheDocument();
    });

    // Check type tags/badges in list view
    const list = screen.getByRole("list", { name: "Graph entities list" });
    expect(within(list).getByText("Model")).toBeInTheDocument();
    expect(within(list).getByText("Metric")).toBeInTheDocument();
    expect(within(list).getByText("Dataset")).toBeInTheDocument();

    // Check pagination controls
    expect(
      screen.getByRole("button", { name: "Previous" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next" })).toBeInTheDocument();
    expect(screen.getByText(/Showing 1–3 of 3/)).toBeInTheDocument();
  });

  // Test 3: Filters nodes by search query and entity type.
  it("filters nodes by search query and entity type", async () => {
    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    // Search input
    const searchInput = screen.getByLabelText("Search entities");
    fireEvent.change(searchInput, { target: { value: "Transformer" } });

    // Entity type dropdown
    const typeSelect = screen.getByLabelText("Filter by entity type");
    fireEvent.change(typeSelect, { target: { value: "Model" } });

    await waitFor(() => {
      expect(searchGraphNodes).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        expect.objectContaining({
          query: "Transformer",
          entityType: "Model",
          skip: 0,
        }),
      );
    });
  });

  // Test 4: Expands 1-hop neighbors on node click.
  it("expands 1-hop neighbors on node click", async () => {
    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    // Switch to list view
    fireEvent.click(screen.getByRole("button", { name: "List View" }));

    await waitFor(() => {
      expect(screen.getByText("Transformer")).toBeInTheDocument();
    });

    // Click node
    fireEvent.click(screen.getByText("Transformer"));

    await waitFor(() => {
      expect(fetchNodeNeighbors).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        "node-transformer",
      );
    });

    // Verify neighbors are displayed
    await waitFor(() => {
      expect(screen.getByText("1-Hop Connections (2)")).toBeInTheDocument();
      // Direction indicators: outgoing (→) and incoming (←)
      expect(screen.getByText("→")).toBeInTheDocument();
      expect(screen.getByText("←")).toBeInTheDocument();
      // Predicate tags
      expect(screen.getByText("EVALUATED_ON")).toBeInTheDocument();
      expect(screen.getByText("PROPOSES_METHOD")).toBeInTheDocument();
    });
  });

  // Test 5: Selecting a neighbor fetches fact detail; renders verified citation and embeds PdfViewer.
  it("selecting a neighbor fetches fact detail; renders verified citation and embeds PdfViewer", async () => {
    // Mock global fetch for paper retrieval
    vi.spyOn(global, "fetch").mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: "paper-123",
        project_id: mockProjectId,
        filename: "Attention Is All You Need.pdf",
        status: "READY",
        document_sha256: "aabbcc112233".padEnd(64, "0"),
      }),
    } as Response);

    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    fireEvent.click(screen.getByRole("button", { name: "List View" }));

    await waitFor(() => {
      expect(screen.getByText("Transformer")).toBeInTheDocument();
    });

    // Expand neighbors
    fireEvent.click(screen.getByText("Transformer"));

    await waitFor(() => {
      expect(screen.getByText("EVALUATED_ON")).toBeInTheDocument();
    });

    // Click on neighbor card
    fireEvent.click(screen.getByText("EVALUATED_ON"));

    await waitFor(() => {
      expect(fetchFactDetail).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        "fact-1",
      );
    });

    // Verify Fact Detail panel contents
    await waitFor(() => {
      expect(screen.getByText("Fact Detail")).toBeInTheDocument();
      expect(screen.getByText("Verified M1 Citation")).toBeInTheDocument();
      expect(screen.getByRole("blockquote")).toHaveTextContent(
        /On the WMT 2014 English-to-German translation task, the big transformer model achieves 28.4 BLEU./,
      );
      expect(screen.getByText("28.4")).toBeInTheDocument();
      expect(screen.getByTestId("mock-pdf-viewer")).toBeInTheDocument();
      const workspaceLink = screen.getByRole("link", {
        name: "Open in Workspace",
      });
      expect(workspaceLink).toHaveAttribute(
        "href",
        `/?project=${mockProjectId}&paper=paper-123&page=6&fact=fact-1`,
      );
    });
  });

  // Test 6: Selecting an unverified fact shows "Exact highlight unavailable" notice with zero guessed rectangle.
  it("selecting an unverified fact shows 'Exact highlight unavailable' notice with zero guessed rectangle", async () => {
    vi.mocked(fetchFactDetail).mockResolvedValueOnce(mockUnverifiedFact);

    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    fireEvent.click(screen.getByRole("button", { name: "List View" }));

    await waitFor(() => {
      expect(screen.getByText("Transformer")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Transformer"));

    await waitFor(() => {
      expect(screen.getByText("PROPOSES_METHOD")).toBeInTheDocument();
    });

    // Click on unverified fact neighbor
    fireEvent.click(screen.getByText("PROPOSES_METHOD"));

    await waitFor(() => {
      expect(fetchFactDetail).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        "fact-2",
      );
    });

    await waitFor(() => {
      // Must show the exact notice
      expect(
        screen.getByText(
          /Exact highlight unavailable — citation anchor could not be verified against the current PDF document. No guessed rectangle shown./,
        ),
      ).toBeInTheDocument();
      // Must NOT render PdfViewer
      expect(screen.queryByTestId("mock-pdf-viewer")).not.toBeInTheDocument();
    });
  });

  // Test 7: Handles Neo4j offline (status neo4j_available: false or 503 error) with graceful outage banner.
  it("handles Neo4j offline with graceful outage banner", async () => {
    vi.mocked(fetchGraphStatus).mockResolvedValueOnce(mockStatusOffline);

    const { unmount } = render(
      <GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />,
    );

    await waitFor(() => {
      expect(screen.getByText("Neo4j Offline")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Graph service is currently offline. Showing local snapshot view.",
        ),
      ).toBeInTheDocument();
    });

    unmount();

    // Also test 503 GraphUnavailableError handling
    vi.mocked(fetchGraphStatus).mockRejectedValueOnce(
      new GraphUnavailableError("Graph service offline"),
    );

    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    await waitFor(() => {
      expect(screen.getByText("Neo4j Offline")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Graph service is currently offline. Showing local snapshot view.",
        ),
      ).toBeInTheDocument();
    });
  });

  // Test 8: Keyboard navigation support (tab, enter to select nodes in list view).
  it("supports keyboard navigation in list view", async () => {
    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    // Switch to List View
    fireEvent.click(screen.getByRole("button", { name: "List View" }));

    await waitFor(() => {
      expect(screen.getByText("Transformer")).toBeInTheDocument();
    });

    const transformerNode = screen
      .getByText("Transformer")
      .closest("[role='button']");
    expect(transformerNode).toBeInTheDocument();
    expect(transformerNode).toHaveAttribute("tabindex", "0");

    // Press Enter to select
    fireEvent.keyDown(transformerNode!, { key: "Enter" });

    await waitFor(() => {
      expect(fetchNodeNeighbors).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        "node-transformer",
      );
    });

    expect(screen.getByText("1-Hop Connections (2)")).toBeInTheDocument();
  });

  // Test 9: Triggers graph indexing on Index Papers button click with confirmation
  it("triggers graph indexing on Index Papers button click with confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    vi.mocked(triggerGraphIndex).mockResolvedValueOnce({
      dry_run: false,
      eligible_paper_ids: ["paper-1"],
      enqueued_count: 3,
      skipped_count: 0,
      target_project_id: mockProjectId,
    });

    render(<GraphExplorer projectId={mockProjectId} apiUrl={mockApiUrl} />);

    const indexBtn = screen.getByRole("button", { name: "Index Papers" });
    fireEvent.click(indexBtn);

    await waitFor(() => {
      expect(window.confirm).toHaveBeenCalledWith(
        "Index papers for this project into the knowledge graph?",
      );
      expect(triggerGraphIndex).toHaveBeenCalledWith(
        mockApiUrl,
        mockProjectId,
        undefined,
        10,
        false,
      );
    });

    await waitFor(() => {
      expect(
        screen.getByText("Enqueued 3 papers for graph indexing."),
      ).toBeInTheDocument();
    });
  });
});
