import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  fetchFactDetail,
  fetchGraphStatus,
  fetchNodeDetail,
  fetchNodeNeighbors,
  fetchRelationships,
  GraphError,
  GraphNotFoundError,
  GraphUnavailableError,
  searchGraphNodes,
  triggerGraphIndex,
} from "./graph";
import type {
  GraphFactDetail,
  GraphIndexResponse,
  GraphNeighbor,
  GraphNode,
  GraphStatus,
} from "@/types";

const mockApiUrl = "http://localhost:8000";
const mockProjectId = "11111111-1111-1111-1111-111111111111";

describe("graph API client", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("fetchGraphStatus", () => {
    it("returns typed graph status for a project", async () => {
      const mockStatus: GraphStatus = {
        project_id: mockProjectId,
        graphrag_enabled: true,
        neo4j_available: true,
        node_count: 42,
        fact_count: 128,
        pending_events_count: 1,
        completed_events_count: 15,
        failed_events_count: 0,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockStatus,
      });
      vi.stubGlobal("fetch", fetchMock);

      const status = await fetchGraphStatus(mockApiUrl, mockProjectId);

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/status`,
        undefined,
      );
      expect(status).toEqual(mockStatus);
    });

    it("handles trailing slash in apiUrl cleanly", async () => {
      const mockStatus: GraphStatus = {
        project_id: mockProjectId,
        graphrag_enabled: false,
        neo4j_available: false,
        node_count: 0,
        fact_count: 0,
        pending_events_count: 0,
        completed_events_count: 0,
        failed_events_count: 0,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockStatus,
      });
      vi.stubGlobal("fetch", fetchMock);

      const status = await fetchGraphStatus(`${mockApiUrl}///`, mockProjectId);

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/status`,
        undefined,
      );
      expect(status).toEqual(mockStatus);
    });
  });

  describe("searchGraphNodes", () => {
    it("passes query parameters and parses pagination response", async () => {
      const mockNodes: GraphNode[] = [
        {
          key: "transformer_model",
          project_id: mockProjectId,
          name: "Transformer",
          type: "Model",
          description: "Attention-based architecture",
          aliases: ["Transformer architecture"],
          updated_at: "2026-09-29T12:00:00Z",
        },
      ];

      const mockResponse = {
        items: mockNodes,
        total: 1,
        limit: 20,
        skip: 10,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await searchGraphNodes(mockApiUrl, mockProjectId, {
        query: "transformer",
        entityType: "Model",
        limit: 20,
        skip: 10,
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const calledUrl = fetchMock.mock.calls[0][0];
      expect(calledUrl).toBe(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/nodes?query=transformer&entity_type=Model&limit=20&skip=10`,
      );
      expect(result).toEqual(mockResponse);
    });

    it("works without search parameters", async () => {
      const mockResponse = {
        items: [],
        total: 0,
        limit: 50,
        skip: 0,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await searchGraphNodes(mockApiUrl, mockProjectId);

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/nodes`,
        undefined,
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe("fetchNodeDetail", () => {
    it("returns node details for valid node key", async () => {
      const mockNode: GraphNode = {
        key: "bert",
        project_id: mockProjectId,
        name: "BERT",
        type: "Model",
        description: "Bidirectional Encoder Representations from Transformers",
        aliases: ["BERT-Base", "BERT-Large"],
        updated_at: "2026-09-29T10:00:00Z",
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockNode,
      });
      vi.stubGlobal("fetch", fetchMock);

      const node = await fetchNodeDetail(mockApiUrl, mockProjectId, "bert");

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/nodes/bert`,
        undefined,
      );
      expect(node).toEqual(mockNode);
    });

    it("throws GraphNotFoundError on 404", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 404,
        statusText: "Not Found",
        json: async () => ({ detail: "Node not found in this project" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(
        fetchNodeDetail(mockApiUrl, mockProjectId, "non_existent"),
      ).rejects.toThrow(GraphNotFoundError);

      await expect(
        fetchNodeDetail(mockApiUrl, mockProjectId, "non_existent"),
      ).rejects.toThrow("Node not found in this project");
    });
  });

  describe("fetchNodeNeighbors", () => {
    it("returns neighbors list with filters applied", async () => {
      const mockNeighbors: GraphNeighbor[] = [
        {
          neighbor_key: "squad_dataset",
          neighbor_name: "SQuAD",
          neighbor_type: "Dataset",
          direction: "OUTGOING",
          predicate: "EVALUATED_ON",
          fact_id: "fact-101",
          qualifiers: { split: "dev", metric: "F1", score: 91.2 },
        },
      ];

      const mockResponse = {
        node_key: "bert",
        neighbors: mockNeighbors,
        total: 1,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await fetchNodeNeighbors(
        mockApiUrl,
        mockProjectId,
        "bert",
        {
          direction: "OUTGOING",
          predicate: "EVALUATED_ON",
          limit: 10,
        },
      );

      const calledUrl = fetchMock.mock.calls[0][0];
      expect(calledUrl).toBe(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/nodes/bert/neighbors?direction=OUTGOING&predicate=EVALUATED_ON&limit=10`,
      );
      expect(result).toEqual(mockResponse);
    });

    it("fetches neighbors without optional filters", async () => {
      const mockResponse = {
        node_key: "bert",
        neighbors: [],
        total: 0,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await fetchNodeNeighbors(
        mockApiUrl,
        mockProjectId,
        "bert",
      );

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/nodes/bert/neighbors`,
        undefined,
      );
      expect(result).toEqual(mockResponse);
    });
  });

  describe("fetchFactDetail", () => {
    it("returns fact with verified citation", async () => {
      const mockFact: GraphFactDetail = {
        id: "fact-001",
        project_id: mockProjectId,
        paper_id: "22222222-2222-2222-2222-222222222222",
        paper_title: "Attention Is All You Need",
        generation_id: "gen-99",
        predicate: "PROPOSES_METHOD",
        subject_key: "attention_paper",
        subject_name: "Attention Is All You Need",
        subject_type: "Paper",
        object_key: "transformer_model",
        object_name: "Transformer",
        object_type: "Model",
        qualifiers: null,
        exact_quote:
          "We propose the Transformer, a model architecture eschewing recurrence.",
        page_number: 1,
        char_start: 120,
        char_end: 191,
        document_sha256: "abcdef123456",
        updated_at: "2026-09-29T10:00:00Z",
        citation: {
          citation_index: 1,
          evidence_id: "G1",
          paper_id: "22222222-2222-2222-2222-222222222222",
          page_number: 1,
          bounding_boxes: [
            {
              x_min: 10,
              y_min: 20,
              x_max: 300,
              y_max: 50,
              page_width: 612,
              page_height: 792,
            },
          ],
          quote:
            "We propose the Transformer, a model architecture eschewing recurrence.",
          anchor_status: "verified",
        },
        anchor_status: "verified",
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockFact,
      });
      vi.stubGlobal("fetch", fetchMock);

      const fact = await fetchFactDetail(mockApiUrl, mockProjectId, "fact-001");

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/facts/fact-001`,
        undefined,
      );
      expect(fact).toEqual(mockFact);
      expect(fact.citation).not.toBeNull();
      expect(fact.anchor_status).toBe("verified");
    });

    it("returns fact with unresolved citation", async () => {
      const mockFact: GraphFactDetail = {
        id: "fact-002",
        project_id: mockProjectId,
        paper_id: "22222222-2222-2222-2222-222222222222",
        paper_title: "Attention Is All You Need",
        generation_id: "gen-99",
        predicate: "EVALUATED_ON",
        subject_key: "transformer_model",
        subject_name: "Transformer",
        subject_type: "Model",
        object_key: "wmt_dataset",
        object_name: "WMT 2014 English-to-German",
        object_type: "Dataset",
        qualifiers: { metric: "BLEU", score: 28.4 },
        exact_quote: "The model achieves 28.4 BLEU on English-to-German.",
        page_number: 8,
        char_start: 500,
        char_end: 556,
        document_sha256: "abcdef123456",
        updated_at: "2026-09-29T10:00:00Z",
        citation: null,
        anchor_status: "unresolved",
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockFact,
      });
      vi.stubGlobal("fetch", fetchMock);

      const fact = await fetchFactDetail(mockApiUrl, mockProjectId, "fact-002");

      expect(fact.citation).toBeNull();
      expect(fact.anchor_status).toBe("unresolved");
    });
  });

  describe("fetchRelationships", () => {
    it("returns relationships list between subject and object", async () => {
      const mockResponse = {
        items: [
          {
            id: "fact-555",
            project_id: mockProjectId,
            paper_id: "22222222-2222-2222-2222-222222222222",
            generation_id: "gen-1",
            predicate: "EVALUATED_ON",
            subject_key: "bert",
            subject_name: "BERT",
            subject_type: "Model",
            object_key: "squad",
            object_name: "SQuAD",
            object_type: "Dataset",
            exact_quote: "BERT achieves 93.2 on SQuAD.",
            page_number: 5,
            char_start: 10,
            char_end: 38,
            document_sha256: "sha256abc",
            anchor_status: "verified" as const,
          },
        ],
        total: 1,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const result = await fetchRelationships(
        mockApiUrl,
        mockProjectId,
        "bert",
        "squad",
        "EVALUATED_ON",
      );

      const calledUrl = fetchMock.mock.calls[0][0];
      expect(calledUrl).toBe(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/relationships?subject_key=bert&object_key=squad&predicate=EVALUATED_ON`,
      );
      expect(result).toEqual(mockResponse);
    });

    it("works without predicate filter", async () => {
      const mockResponse = { items: [], total: 0 };
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      await fetchRelationships(mockApiUrl, mockProjectId, "bert", "squad");

      const calledUrl = fetchMock.mock.calls[0][0];
      expect(calledUrl).toBe(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/relationships?subject_key=bert&object_key=squad`,
      );
    });
  });

  describe("triggerGraphIndex", () => {
    it("sends dry_run payload and returns response", async () => {
      const mockResponse: GraphIndexResponse = {
        dry_run: true,
        eligible_paper_ids: ["paper-1", "paper-2"],
        enqueued_count: 2,
        skipped_count: 0,
        target_project_id: mockProjectId,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const res = await triggerGraphIndex(mockApiUrl, mockProjectId, {
        paperIds: ["paper-1", "paper-2"],
        limit: 5,
        dryRun: true,
      });

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/index`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            paper_ids: ["paper-1", "paper-2"],
            limit: 5,
            dry_run: true,
          }),
        },
      );
      expect(res).toEqual(mockResponse);
    });

    it("triggers index without payload using defaults", async () => {
      const mockResponse: GraphIndexResponse = {
        dry_run: true,
        eligible_paper_ids: [],
        enqueued_count: 0,
        skipped_count: 0,
        target_project_id: mockProjectId,
      };

      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        json: async () => mockResponse,
      });
      vi.stubGlobal("fetch", fetchMock);

      const res = await triggerGraphIndex(mockApiUrl, mockProjectId);

      expect(fetchMock).toHaveBeenCalledWith(
        `${mockApiUrl}/api/v1/projects/${mockProjectId}/graph/index`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: "{}",
        },
      );
      expect(res).toEqual(mockResponse);
    });
  });

  describe("outage / 503 handling", () => {
    it("raises GraphUnavailableError when Neo4j or graph service is offline (503)", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
        json: async () => ({ detail: "Graph service unavailable" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(searchGraphNodes(mockApiUrl, mockProjectId)).rejects.toThrow(
        GraphUnavailableError,
      );

      await expect(searchGraphNodes(mockApiUrl, mockProjectId)).rejects.toThrow(
        "Graph service unavailable",
      );

      // Verify status and inheritance
      try {
        await searchGraphNodes(mockApiUrl, mockProjectId);
      } catch (err) {
        expect(err).toBeInstanceOf(GraphUnavailableError);
        expect(err).toBeInstanceOf(GraphError);
        expect((err as GraphUnavailableError).status).toBe(503);
      }
    });

    it("handles 503 with default message when body is empty", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        statusText: "Service Unavailable",
        json: async () => {
          throw new Error("No JSON");
        },
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(
        fetchNodeNeighbors(mockApiUrl, mockProjectId, "any-node"),
      ).rejects.toThrow(GraphUnavailableError);
    });
  });

  describe("generic error & network error handling", () => {
    it("raises GraphError on general HTTP 500 error", async () => {
      const fetchMock = vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        statusText: "Internal Server Error",
        json: async () => ({ detail: "Database connection failed" }),
      });
      vi.stubGlobal("fetch", fetchMock);

      await expect(fetchGraphStatus(mockApiUrl, mockProjectId)).rejects.toThrow(
        GraphError,
      );

      try {
        await fetchGraphStatus(mockApiUrl, mockProjectId);
      } catch (err) {
        expect(err).toBeInstanceOf(GraphError);
        expect((err as GraphError).status).toBe(500);
        expect((err as GraphError).message).toBe("Database connection failed");
      }
    });

    it("raises GraphError with network error description when fetch fails", async () => {
      const fetchMock = vi.fn().mockRejectedValue(new Error("Failed to fetch"));
      vi.stubGlobal("fetch", fetchMock);

      await expect(fetchGraphStatus(mockApiUrl, mockProjectId)).rejects.toThrow(
        GraphError,
      );

      await expect(fetchGraphStatus(mockApiUrl, mockProjectId)).rejects.toThrow(
        "Network error: Failed to fetch",
      );
    });
  });
});
