"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import PdfViewer from "@/components/PdfViewer";
import {
  fetchFactDetail,
  fetchGraphStatus,
  fetchNodeDetail,
  fetchNodeNeighbors,
  searchGraphNodes,
  triggerGraphIndex,
  GraphUnavailableError,
} from "@/lib/graph";
import type {
  EntityType,
  GraphFactDetail,
  GraphNeighbor,
  GraphNode,
  GraphStatus,
  Paper,
} from "@/types";

export interface GraphExplorerProps {
  projectId: string;
  apiUrl: string;
  initialNodeKey?: string | null;
  initialFactId?: string | null;
}

const ENTITY_TYPES: EntityType[] = [
  "Paper",
  "Author",
  "Institution",
  "Task",
  "Method",
  "Model",
  "Dataset",
  "Metric",
  "Result",
  "Claim",
  "Limitation",
  "Concept",
];

export default function GraphExplorer({
  projectId,
  apiUrl,
  initialNodeKey = null,
  initialFactId = null,
}: GraphExplorerProps) {
  // Status and metrics
  const [status, setStatus] = useState<GraphStatus | null>(null);
  const [isOffline, setIsOffline] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);

  // Filters & view
  const [viewMode, setViewMode] = useState<"visual" | "list">("visual");
  const [searchQuery, setSearchQuery] = useState("");
  const [entityTypeFilter, setEntityTypeFilter] = useState<string>("");

  // Nodes & pagination
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [totalNodes, setTotalNodes] = useState(0);
  const [skip, setSkip] = useState(0);
  const limit = 10;
  const [isLoadingNodes, setIsLoadingNodes] = useState(false);
  const [nodesError, setNodesError] = useState<string | null>(null);

  // Selected Node & Neighbors
  const [selectedNodeKey, setSelectedNodeKey] = useState<string | null>(
    initialNodeKey,
  );
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [neighbors, setNeighbors] = useState<GraphNeighbor[]>([]);
  const [isLoadingNeighbors, setIsLoadingNeighbors] = useState(false);
  const [neighborsError, setNeighborsError] = useState<string | null>(null);

  // Selected Fact Detail & Source Jump
  const [selectedFactId, setSelectedFactId] = useState<string | null>(
    initialFactId,
  );
  const [selectedFact, setSelectedFact] = useState<GraphFactDetail | null>(
    null,
  );
  const [isLoadingFact, setIsLoadingFact] = useState(false);
  const [factError, setFactError] = useState<string | null>(null);
  const [loadedPaper, setLoadedPaper] = useState<Paper | null>(null);
  const factRequestId = useRef(0);

  // Visual graph pan & zoom state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });

  // 1. Fetch Status
  const loadStatus = useCallback(async () => {
    try {
      const data = await fetchGraphStatus(apiUrl, projectId);
      setStatus(data);
      setIsOffline(!data.neo4j_available);
    } catch (err) {
      if (err instanceof GraphUnavailableError) {
        setIsOffline(true);
      }
      setStatus(null);
    }
  }, [apiUrl, projectId]);

  useEffect(() => {
    let ignore = false;
    async function fetchStatus() {
      try {
        const data = await fetchGraphStatus(apiUrl, projectId);
        if (!ignore) {
          setStatus(data);
          setIsOffline(!data.neo4j_available);
        }
      } catch (err) {
        if (!ignore) {
          if (err instanceof GraphUnavailableError) {
            setIsOffline(true);
          }
          setStatus(null);
        }
      }
    }
    fetchStatus();
    return () => {
      ignore = true;
    };
  }, [apiUrl, projectId]);

  // 2. Search Nodes
  const loadNodes = useCallback(
    async (queryText: string, typeText: string, currentSkip: number) => {
      setIsLoadingNodes(true);
      setNodesError(null);
      try {
        const res = await searchGraphNodes(apiUrl, projectId, {
          query: queryText,
          entityType: typeText,
          skip: currentSkip,
          limit,
        });
        setNodes(res.items || []);
        setTotalNodes(res.total || 0);
      } catch (err) {
        if (err instanceof GraphUnavailableError) {
          setIsOffline(true);
        }
        setNodes([]);
        setTotalNodes(0);
        setNodesError(
          err instanceof Error ? err.message : "Failed to load graph nodes.",
        );
      } finally {
        setIsLoadingNodes(false);
      }
    },
    [apiUrl, projectId, limit],
  );

  useEffect(() => {
    let ignore = false;
    async function fetchNodes() {
      try {
        const res = await searchGraphNodes(apiUrl, projectId, {
          query: searchQuery,
          entityType: entityTypeFilter,
          skip,
          limit,
        });
        if (!ignore) {
          setNodes(res.items || []);
          setTotalNodes(res.total || 0);
          setNodesError(null);
        }
      } catch (err) {
        if (!ignore) {
          if (err instanceof GraphUnavailableError) {
            setIsOffline(true);
          }
          setNodes([]);
          setTotalNodes(0);
          setNodesError(
            err instanceof Error ? err.message : "Failed to load graph nodes.",
          );
        }
      }
    }
    fetchNodes();
    return () => {
      ignore = true;
    };
  }, [apiUrl, projectId, searchQuery, entityTypeFilter, skip, limit]);

  // 3. Load Neighbors when a node is selected
  const loadNeighbors = useCallback(
    async (nodeKey: string) => {
      setIsLoadingNeighbors(true);
      setNeighborsError(null);
      try {
        const res = await fetchNodeNeighbors(apiUrl, projectId, nodeKey);
        setNeighbors(res.neighbors || []);
      } catch (err) {
        if (err instanceof GraphUnavailableError) {
          setIsOffline(true);
        }
        setNeighbors([]);
        setNeighborsError(
          err instanceof Error
            ? err.message
            : "Failed to load connected neighbors.",
        );
      } finally {
        setIsLoadingNeighbors(false);
      }
    },
    [apiUrl, projectId],
  );

  const handleSelectNode = useCallback(
    (node: GraphNode) => {
      setSelectedNode(node);
      setSelectedNodeKey(node.key);
      loadNeighbors(node.key);
    },
    [loadNeighbors],
  );

  // If initialNodeKey is supplied, find or fetch it
  useEffect(() => {
    if (!initialNodeKey) return;
    const nodeKey = initialNodeKey;
    let ignore = false;
    async function initNode() {
      try {
        const res = await fetchNodeNeighbors(apiUrl, projectId, nodeKey);
        if (!ignore) {
          const match = nodes.find((n) => n.key === nodeKey);
          if (match) {
            setSelectedNode(match);
          } else {
            fetchNodeDetail(apiUrl, projectId, nodeKey)
              .then((node) => {
                if (!ignore) setSelectedNode(node);
              })
              .catch(() => {});
          }
          setSelectedNodeKey(nodeKey);
          setNeighbors(res.neighbors || []);
        }
      } catch {
        // ignore
      }
    }
    initNode();
    return () => {
      ignore = true;
    };
  }, [initialNodeKey, nodes, apiUrl, projectId]);

  // 4. Load Fact Detail when a fact is selected
  const loadFact = useCallback(
    async (factId: string) => {
      const requestId = ++factRequestId.current;
      setIsLoadingFact(true);
      setFactError(null);
      setSelectedFact(null);
      setLoadedPaper(null);
      try {
        const fact = await fetchFactDetail(apiUrl, projectId, factId);
        let paper: Paper | null = null;
        if (fact.anchor_status === "verified" && fact.citation) {
          try {
            const paperRes = await fetch(
              `${apiUrl}/api/v1/papers/${fact.paper_id}`,
            );
            if (paperRes.ok) {
              paper = (await paperRes.json()) as Paper;
            } else {
              paper = {
                id: fact.paper_id,
                project_id: projectId,
                filename: fact.paper_title || "Paper",
                status: "READY",
                document_sha256: fact.document_sha256,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
              };
            }
          } catch {
            paper = {
              id: fact.paper_id,
              project_id: projectId,
              filename: fact.paper_title || "Paper",
              status: "READY",
              document_sha256: fact.document_sha256,
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString(),
            };
          }
        }
        if (factRequestId.current === requestId) {
          setSelectedFactId(factId);
          setSelectedFact(fact);
          setLoadedPaper(paper);
        }
      } catch (err) {
        if (factRequestId.current === requestId) {
          if (err instanceof GraphUnavailableError) {
            setIsOffline(true);
          }
          setFactError(
            err instanceof Error ? err.message : "Failed to load fact details.",
          );
        }
      } finally {
        if (factRequestId.current === requestId) setIsLoadingFact(false);
      }
    },
    [apiUrl, projectId],
  );

  const handleSelectNeighbor = (neighbor: GraphNeighbor) => {
    setSelectedFactId(neighbor.fact_id);
    loadFact(neighbor.fact_id);
  };

  useEffect(() => {
    if (!initialFactId) return;
    const timer = window.setTimeout(() => void loadFact(initialFactId), 0);
    return () => {
      window.clearTimeout(timer);
      factRequestId.current += 1;
    };
  }, [initialFactId, loadFact]);

  // 5. Index Papers Action
  const handleIndexPapers = async () => {
    const confirmed = window.confirm(
      "Index papers for this project into the knowledge graph?",
    );
    if (!confirmed) return;

    setIsIndexing(true);
    setToastMessage(null);
    try {
      const res = await triggerGraphIndex(
        apiUrl,
        projectId,
        undefined,
        10,
        false,
      );
      setToastMessage(
        `Enqueued ${res.enqueued_count} papers for graph indexing.`,
      );
      loadStatus();
      loadNodes(searchQuery, entityTypeFilter, skip);
    } catch (err) {
      setToastMessage(
        `Indexing failed: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsIndexing(false);
    }
  };

  // Helper for entity badge colors
  const getTypeBadgeClass = (type: string) => {
    switch (type) {
      case "Paper":
        return "bg-blue-100 text-blue-800 border-blue-200";
      case "Method":
        return "bg-purple-100 text-purple-800 border-purple-200";
      case "Model":
        return "bg-indigo-100 text-indigo-800 border-indigo-200";
      case "Dataset":
        return "bg-teal-100 text-teal-800 border-teal-200";
      case "Metric":
        return "bg-pink-100 text-pink-800 border-pink-200";
      case "Result":
        return "bg-orange-100 text-orange-800 border-orange-200";
      case "Claim":
        return "bg-amber-100 text-amber-800 border-amber-200";
      case "Limitation":
        return "bg-rose-100 text-rose-800 border-rose-200";
      case "Author":
        return "bg-cyan-100 text-cyan-800 border-cyan-200";
      case "Institution":
        return "bg-sky-100 text-sky-800 border-sky-200";
      case "Task":
        return "bg-lime-100 text-lime-800 border-lime-200";
      default:
        return "bg-zinc-100 text-zinc-800 border-zinc-200";
    }
  };

  const entityCount = status ? status.node_count : totalNodes;
  const factCount = status ? status.fact_count : 0;
  const totalPages = Math.ceil(totalNodes / limit);
  const currentPage = Math.floor(skip / limit) + 1;

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Top Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs">
        <div className="flex flex-wrap items-center gap-3">
          {/* Status Badge */}
          {!isOffline && status?.neo4j_available ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-xs font-semibold text-emerald-800">
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              Neo4j Connected
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-50 px-3 py-1 text-xs font-semibold text-amber-800">
              <span className="h-2 w-2 rounded-full bg-amber-500" />
              Neo4j Offline
            </span>
          )}

          {/* Metrics */}
          <div className="flex items-center gap-4 text-xs text-zinc-600 pl-2 border-l border-zinc-200">
            <span>
              Entity Count:{" "}
              <strong className="font-semibold text-zinc-900">
                {entityCount}
              </strong>
            </span>
            <span>
              Fact Count:{" "}
              <strong className="font-semibold text-zinc-900">
                {factCount}
              </strong>
            </span>
          </div>
        </div>

        {/* Action Button: Index Papers */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleIndexPapers}
            disabled={isIndexing}
            className="inline-flex items-center justify-center rounded-lg bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white shadow-2xs hover:bg-zinc-800 disabled:opacity-50 transition"
          >
            {isIndexing ? "Indexing..." : "Index Papers"}
          </button>
        </div>
      </div>

      {/* Outage Notice if Neo4j is offline */}
      {(isOffline || (status && !status.neo4j_available)) && (
        <div
          role="alert"
          className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-xs text-amber-900"
        >
          <div className="flex items-center gap-2 font-semibold">
            <span>⚠️</span>
            <span>
              Graph service is currently offline. Showing local snapshot view.
            </span>
          </div>
        </div>
      )}

      {/* Toast / Action Message */}
      {toastMessage && (
        <div className="flex items-center justify-between rounded-lg border border-blue-200 bg-blue-50 px-4 py-2 text-xs text-blue-800">
          <span>{toastMessage}</span>
          <button
            type="button"
            onClick={() => setToastMessage(null)}
            className="text-blue-600 hover:text-blue-900 ml-4 font-bold"
          >
            ✕
          </button>
        </div>
      )}

      {/* Search & Filters & View Toggle Bar */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs">
        <div className="flex flex-1 flex-wrap items-center gap-3 min-w-[280px]">
          {/* Search Input */}
          <div className="relative flex-1 min-w-[200px]">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                const val = e.target.value;
                setSearchQuery(val);
                setSkip(0);
              }}
              placeholder="Search entities by name..."
              aria-label="Search entities"
              className="w-full rounded-lg border border-zinc-300 px-3 py-1.5 text-xs text-zinc-900 placeholder-zinc-400 focus:border-zinc-900 focus:outline-hidden"
            />
          </div>

          {/* Entity Type Filter */}
          <div className="flex items-center gap-1.5">
            <label
              htmlFor="entity-type-filter"
              className="text-xs text-zinc-500 whitespace-nowrap"
            >
              Type:
            </label>
            <select
              id="entity-type-filter"
              value={entityTypeFilter}
              onChange={(e) => {
                const val = e.target.value;
                setEntityTypeFilter(val);
                setSkip(0);
              }}
              aria-label="Filter by entity type"
              className="rounded-lg border border-zinc-300 bg-white px-2 py-1.5 text-xs text-zinc-900 focus:border-zinc-900 focus:outline-hidden"
            >
              <option value="">All Types</option>
              {ENTITY_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* View Mode Toggle */}
        <div className="flex items-center gap-1 rounded-lg bg-zinc-100 p-1">
          <button
            type="button"
            onClick={() => setViewMode("visual")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
              viewMode === "visual"
                ? "bg-white text-zinc-950 shadow-2xs"
                : "text-zinc-600 hover:text-zinc-950"
            }`}
          >
            Visual Graph
          </button>
          <button
            type="button"
            onClick={() => setViewMode("list")}
            className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
              viewMode === "list"
                ? "bg-white text-zinc-950 shadow-2xs"
                : "text-zinc-600 hover:text-zinc-950"
            }`}
          >
            List View
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 min-h-[500px]">
        {/* Left/Main Column: Visual Graph or List View */}
        <div
          className={`flex flex-col gap-4 ${
            selectedFact
              ? "lg:col-span-4"
              : selectedNode
                ? "lg:col-span-7"
                : "lg:col-span-12"
          }`}
        >
          {isLoadingNodes ? (
            <div className="flex flex-1 items-center justify-center rounded-xl border border-zinc-200 bg-white p-12 text-xs text-zinc-500 shadow-2xs">
              Loading graph nodes…
            </div>
          ) : nodesError ? (
            <div className="flex flex-1 items-center justify-center rounded-xl border border-red-200 bg-red-50 p-12 text-xs text-red-700 shadow-2xs">
              {nodesError}
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex flex-1 items-center justify-center rounded-xl border border-zinc-200 bg-white p-12 text-xs text-zinc-500 shadow-2xs">
              No entities found matching the selected filters.
            </div>
          ) : viewMode === "list" ? (
            /* List View */
            <div className="flex flex-col rounded-xl border border-zinc-200 bg-white shadow-2xs overflow-hidden">
              <div
                role="list"
                aria-label="Graph entities list"
                className="divide-y divide-zinc-200"
              >
                {nodes.map((node) => {
                  const isSelected = selectedNodeKey === node.key;
                  return (
                    <div
                      key={node.key}
                      role="button"
                      tabIndex={0}
                      onClick={() => handleSelectNode(node)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleSelectNode(node);
                        }
                      }}
                      className={`flex flex-col gap-1.5 p-4 text-left cursor-pointer transition focus:outline-hidden focus:ring-2 focus:ring-zinc-900 ${
                        isSelected
                          ? "bg-zinc-100 ring-1 ring-zinc-400"
                          : "hover:bg-zinc-50"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-semibold text-sm text-zinc-950">
                          {node.name}
                        </span>
                        <span
                          className={`rounded-md border px-2 py-0.5 text-[11px] font-medium ${getTypeBadgeClass(
                            node.type,
                          )}`}
                        >
                          {node.type}
                        </span>
                      </div>
                      {node.description && (
                        <p className="text-xs text-zinc-600 line-clamp-2">
                          {node.description}
                        </p>
                      )}
                      {node.aliases && node.aliases.length > 0 && (
                        <div className="text-[11px] text-zinc-500">
                          <span className="font-medium">Aliases: </span>
                          {node.aliases.join(", ")}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Pagination */}
              <div className="flex items-center justify-between border-t border-zinc-200 bg-zinc-50 px-4 py-3 text-xs text-zinc-600">
                <span>
                  Showing {skip + 1}–{Math.min(skip + limit, totalNodes)} of{" "}
                  {totalNodes}
                </span>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSkip((s) => Math.max(0, s - limit))}
                    disabled={skip === 0}
                    className="rounded border border-zinc-300 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40 transition"
                  >
                    Previous
                  </button>
                  <span className="text-[11px]">
                    Page {currentPage} of {totalPages || 1}
                  </span>
                  <button
                    type="button"
                    onClick={() => setSkip((s) => s + limit)}
                    disabled={
                      nodes.length < limit ||
                      (totalNodes > 0 && skip + limit >= totalNodes)
                    }
                    className="rounded border border-zinc-300 bg-white px-2.5 py-1 text-xs font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-40 transition"
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* Visual SVG Graph View */
            <div className="relative flex flex-1 flex-col rounded-xl border border-zinc-200 bg-white shadow-2xs overflow-hidden min-h-[480px]">
              {/* Pan & Zoom Controls */}
              <div className="absolute top-3 right-3 z-10 flex items-center gap-1 rounded-lg border border-zinc-200 bg-white/90 p-1 shadow-2xs backdrop-blur-xs text-xs">
                <button
                  type="button"
                  onClick={() => setZoom((z) => Math.min(2.5, z + 0.2))}
                  className="rounded px-2 py-1 font-bold text-zinc-700 hover:bg-zinc-100"
                  title="Zoom In"
                  aria-label="Zoom in"
                >
                  +
                </button>
                <span className="px-1 font-mono text-[11px] text-zinc-600">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  type="button"
                  onClick={() => setZoom((z) => Math.max(0.4, z - 0.2))}
                  className="rounded px-2 py-1 font-bold text-zinc-700 hover:bg-zinc-100"
                  title="Zoom Out"
                  aria-label="Zoom out"
                >
                  -
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setZoom(1);
                    setPan({ x: 0, y: 0 });
                  }}
                  className="rounded px-1.5 py-1 text-[11px] text-zinc-600 hover:bg-zinc-100"
                  title="Reset View"
                  aria-label="Reset view"
                >
                  ↺
                </button>
              </div>

              {/* Interactive SVG Canvas */}
              <svg
                className="h-full w-full select-none cursor-grab active:cursor-grabbing min-h-[450px]"
                viewBox="0 0 800 600"
                aria-label="Interactive visual knowledge graph"
              >
                <defs>
                  <marker
                    id="arrow-head"
                    viewBox="0 0 10 10"
                    refX="22"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#71717a" />
                  </marker>
                </defs>

                <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                  {selectedNode ? (
                    /* Layout: Selected central node with radiating neighbors */
                    <g>
                      {/* Radiating Neighbor Edges and Nodes */}
                      {neighbors.map((neighbor, i) => {
                        const total = neighbors.length;
                        const angle = (i * 2 * Math.PI) / total;
                        const radius = Math.min(240, Math.max(160, total * 22));
                        const nx = 400 + radius * Math.cos(angle);
                        const ny = 300 + radius * Math.sin(angle);
                        const isSelectedFact =
                          selectedFactId === neighbor.fact_id;

                        return (
                          <g key={`${neighbor.fact_id}-${i}`}>
                            {/* Edge Line */}
                            <line
                              x1={400}
                              y1={300}
                              x2={nx}
                              y2={ny}
                              stroke={isSelectedFact ? "#18181b" : "#d4d4d8"}
                              strokeWidth={isSelectedFact ? 2.5 : 1.5}
                              strokeDasharray={
                                neighbor.direction === "INCOMING"
                                  ? "4 2"
                                  : undefined
                              }
                              markerEnd="url(#arrow-head)"
                            />
                            {/* Predicate label along line */}
                            <rect
                              x={(400 + nx) / 2 - 35}
                              y={(300 + ny) / 2 - 9}
                              width={70}
                              height={18}
                              rx={4}
                              fill="#ffffff"
                              stroke="#e4e4e7"
                              strokeWidth={1}
                            />
                            <text
                              x={(400 + nx) / 2}
                              y={(300 + ny) / 2 + 3}
                              textAnchor="middle"
                              fontSize="9"
                              fill="#52525b"
                              className="pointer-events-none font-mono"
                            >
                              {neighbor.direction === "INCOMING" ? "← " : "→ "}
                              {neighbor.predicate.length > 12
                                ? `${neighbor.predicate.slice(0, 10)}…`
                                : neighbor.predicate}
                            </text>

                            {/* Neighbor Node Circle / Card */}
                            <g
                              role="button"
                              tabIndex={0}
                              onClick={() => handleSelectNeighbor(neighbor)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  handleSelectNeighbor(neighbor);
                                }
                              }}
                              className="cursor-pointer focus:outline-hidden"
                            >
                              <circle
                                cx={nx}
                                cy={ny}
                                r={24}
                                fill={isSelectedFact ? "#f4f4f5" : "#ffffff"}
                                stroke={isSelectedFact ? "#09090b" : "#a1a1aa"}
                                strokeWidth={isSelectedFact ? 2.5 : 1.5}
                              />
                              <text
                                x={nx}
                                y={ny + 3}
                                textAnchor="middle"
                                fontSize="10"
                                fontWeight="bold"
                                fill="#09090b"
                                className="pointer-events-none"
                              >
                                {neighbor.neighbor_name.slice(0, 5)}…
                              </text>
                              <text
                                x={nx}
                                y={ny + 35}
                                textAnchor="middle"
                                fontSize="11"
                                fontWeight="500"
                                fill="#27272a"
                              >
                                {neighbor.neighbor_name.length > 16
                                  ? `${neighbor.neighbor_name.slice(0, 14)}…`
                                  : neighbor.neighbor_name}
                              </text>
                              <text
                                x={nx}
                                y={ny + 48}
                                textAnchor="middle"
                                fontSize="9"
                                fill="#71717a"
                              >
                                ({neighbor.neighbor_type})
                              </text>
                            </g>
                          </g>
                        );
                      })}

                      {/* Central Selected Node */}
                      <g className="cursor-default">
                        <circle
                          cx={400}
                          cy={300}
                          r={34}
                          fill="#18181b"
                          stroke="#3f3f46"
                          strokeWidth={2}
                        />
                        <text
                          x={400}
                          y={298}
                          textAnchor="middle"
                          fontSize="11"
                          fontWeight="bold"
                          fill="#ffffff"
                        >
                          {selectedNode.name.slice(0, 8)}…
                        </text>
                        <text
                          x={400}
                          y={312}
                          textAnchor="middle"
                          fontSize="9"
                          fill="#a1a1aa"
                        >
                          {selectedNode.type}
                        </text>
                        <text
                          x={400}
                          y={350}
                          textAnchor="middle"
                          fontSize="12"
                          fontWeight="bold"
                          fill="#09090b"
                        >
                          {selectedNode.name}
                        </text>
                      </g>
                    </g>
                  ) : (
                    /* Layout: Grid/ring of current page nodes */
                    <g>
                      {nodes.map((node, i) => {
                        const total = nodes.length;
                        const angle = (i * 2 * Math.PI) / total;
                        const radius = 180;
                        const cx = 400 + radius * Math.cos(angle);
                        const cy = 300 + radius * Math.sin(angle);

                        return (
                          <g
                            key={node.key}
                            role="button"
                            tabIndex={0}
                            onClick={() => handleSelectNode(node)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                handleSelectNode(node);
                              }
                            }}
                            className="cursor-pointer focus:outline-hidden"
                          >
                            <circle
                              cx={cx}
                              cy={cy}
                              r={26}
                              fill="#f4f4f5"
                              stroke="#71717a"
                              strokeWidth={1.5}
                              className="hover:fill-zinc-200 transition"
                            />
                            <text
                              x={cx}
                              y={cy + 3}
                              textAnchor="middle"
                              fontSize="10"
                              fontWeight="600"
                              fill="#18181b"
                              className="pointer-events-none"
                            >
                              {node.name.slice(0, 6)}…
                            </text>
                            <text
                              x={cx}
                              y={cy + 38}
                              textAnchor="middle"
                              fontSize="11"
                              fontWeight="500"
                              fill="#27272a"
                            >
                              {node.name.length > 16
                                ? `${node.name.slice(0, 14)}…`
                                : node.name}
                            </text>
                            <text
                              x={cx}
                              y={cy + 50}
                              textAnchor="middle"
                              fontSize="9"
                              fill="#71717a"
                            >
                              {node.type}
                            </text>
                          </g>
                        );
                      })}
                      <text
                        x={400}
                        y={300}
                        textAnchor="middle"
                        fontSize="12"
                        fill="#a1a1aa"
                      >
                        Click any node to explore connections
                      </text>
                    </g>
                  )}
                </g>
              </svg>
            </div>
          )}
        </div>

        {/* Middle Column: 1-Hop Neighbors Panel (When a Node is Selected) */}
        {selectedNode && (
          <div
            className={`flex flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-4 shadow-2xs ${
              selectedFact ? "lg:col-span-3" : "lg:col-span-5"
            }`}
          >
            <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
              <div>
                <h3 className="font-semibold text-sm text-zinc-950">
                  {selectedNode.name}
                </h3>
                <span
                  className={`mt-1 inline-block rounded-md border px-2 py-0.5 text-[10px] font-medium ${getTypeBadgeClass(
                    selectedNode.type,
                  )}`}
                >
                  {selectedNode.type}
                </span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setSelectedNode(null);
                  setSelectedNodeKey(null);
                  setNeighbors([]);
                }}
                className="text-xs text-zinc-500 hover:text-zinc-800"
                aria-label="Clear selected node"
              >
                ✕ Close
              </button>
            </div>

            {/* Neighbors List */}
            <div className="flex flex-col gap-2">
              <h4 className="text-xs font-semibold text-zinc-700">
                1-Hop Connections ({neighbors.length})
              </h4>

              {isLoadingNeighbors ? (
                <div className="p-6 text-center text-xs text-zinc-500">
                  Loading connections…
                </div>
              ) : neighborsError ? (
                <div className="p-4 text-xs text-red-600 bg-red-50 rounded-lg">
                  {neighborsError}
                </div>
              ) : neighbors.length === 0 ? (
                <div className="p-6 text-center text-xs text-zinc-500">
                  No connected neighbors found for this entity.
                </div>
              ) : (
                <div
                  role="list"
                  aria-label="Node neighbors list"
                  className="flex flex-col gap-2 max-h-[500px] overflow-y-auto pr-1"
                >
                  {neighbors.map((neighbor, idx) => {
                    const isFactActive = selectedFactId === neighbor.fact_id;
                    const dirSymbol =
                      neighbor.direction === "OUTGOING"
                        ? "→"
                        : neighbor.direction === "INCOMING"
                          ? "←"
                          : "↔";

                    return (
                      <button
                        key={`${neighbor.fact_id}-${idx}`}
                        type="button"
                        onClick={() => handleSelectNeighbor(neighbor)}
                        className={`flex flex-col gap-1.5 rounded-lg border p-3 text-left transition focus:outline-hidden focus:ring-2 focus:ring-zinc-900 ${
                          isFactActive
                            ? "border-zinc-900 bg-zinc-50 shadow-2xs"
                            : "border-zinc-200 bg-white hover:bg-zinc-50"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-1 text-xs">
                          <span className="font-semibold text-zinc-900 flex items-center gap-1.5">
                            <span className="font-mono text-zinc-500 font-bold">
                              {dirSymbol}
                            </span>
                            <span>{neighbor.neighbor_name}</span>
                          </span>
                          <span
                            className={`rounded-md border px-1.5 py-0.2 text-[10px] ${getTypeBadgeClass(
                              neighbor.neighbor_type,
                            )}`}
                          >
                            {neighbor.neighbor_type}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <span className="rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[10px] text-zinc-700 font-medium">
                            {neighbor.predicate}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Right Column: Fact Detail & Source Jump (Task 5.32) */}
        {(selectedFact || isLoadingFact || factError) && (
          <div className="flex flex-col gap-4 rounded-xl border border-zinc-200 bg-white p-5 shadow-2xs lg:col-span-5">
            <div className="flex items-center justify-between border-b border-zinc-200 pb-3">
              <h3 className="font-bold text-sm text-zinc-950">Fact Detail</h3>
              <button
                type="button"
                onClick={() => {
                  setSelectedFact(null);
                  setSelectedFactId(null);
                  setLoadedPaper(null);
                }}
                className="text-xs text-zinc-500 hover:text-zinc-800"
                aria-label="Close fact detail"
              >
                ✕ Close
              </button>
            </div>

            {isLoadingFact ? (
              <div className="p-8 text-center text-xs text-zinc-500">
                Loading fact details…
              </div>
            ) : factError ? (
              <div className="p-4 text-xs text-red-600 bg-red-50 rounded-lg">
                {factError}
              </div>
            ) : selectedFact ? (
              <div className="flex flex-col gap-4">
                {/* Subject - Predicate - Object */}
                <div className="rounded-lg bg-zinc-50 p-3 text-xs flex flex-col gap-1.5 border border-zinc-200">
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500 font-medium w-16 shrink-0">
                      Subject:
                    </span>
                    <span className="font-semibold text-zinc-900">
                      {selectedFact.subject_name}
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      ({selectedFact.subject_type})
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500 font-medium w-16 shrink-0">
                      Predicate:
                    </span>
                    <span className="font-mono font-semibold text-zinc-800 bg-zinc-200/70 px-1.5 py-0.5 rounded text-[11px]">
                      {selectedFact.predicate}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500 font-medium w-16 shrink-0">
                      Object:
                    </span>
                    <span className="font-semibold text-zinc-900">
                      {selectedFact.object_name}
                    </span>
                    <span className="text-[10px] text-zinc-500">
                      ({selectedFact.object_type})
                    </span>
                  </div>
                </div>

                {/* Qualifiers */}
                {selectedFact.qualifiers &&
                  Object.keys(selectedFact.qualifiers).length > 0 && (
                    <div className="flex flex-col gap-1 text-xs">
                      <span className="font-semibold text-zinc-700">
                        Qualifiers:
                      </span>
                      <div className="grid grid-cols-2 gap-2 rounded-lg bg-zinc-50 p-2.5 border border-zinc-200 text-[11px]">
                        {Object.entries(selectedFact.qualifiers).map(
                          ([k, v]) => (
                            <div key={k} className="flex flex-col">
                              <span className="text-zinc-500 uppercase text-[9px] font-bold">
                                {k}
                              </span>
                              <span className="text-zinc-800 font-medium">
                                {String(v)}
                              </span>
                            </div>
                          ),
                        )}
                      </div>
                    </div>
                  )}

                {/* Provenance Display */}
                <div className="flex flex-col gap-2">
                  <span className="font-semibold text-xs text-zinc-700">
                    Source Provenance:
                  </span>
                  <blockquote className="border-l-4 border-zinc-400 bg-zinc-50/70 pl-3 py-2 italic text-xs text-zinc-800">
                    &ldquo;{selectedFact.exact_quote}&rdquo;
                  </blockquote>

                  <div className="flex flex-col gap-1 text-[11px] text-zinc-600">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-500">Paper:</span>
                      <span>{selectedFact.paper_title || "Paper"}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-500">Page:</span>
                      <span>Page {selectedFact.page_number}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-zinc-500">
                        SHA-256:
                      </span>
                      <code className="font-mono text-[10px] text-zinc-700 truncate">
                        {selectedFact.document_sha256}
                      </code>
                    </div>
                  </div>
                </div>

                {/* Verification Badge & PDF Highlight / Unverified Notice */}
                {selectedFact.anchor_status === "verified" &&
                selectedFact.citation ? (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center gap-1 rounded-full border border-emerald-300 bg-emerald-50 px-2.5 py-0.5 text-xs font-semibold text-emerald-800">
                        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                        Verified M1 Citation
                      </span>
                    </div>

                    {/* Embedded PdfViewer with DOM Range text highlight */}
                    <div className="h-[400px] rounded-lg border border-zinc-200 overflow-hidden">
                      <PdfViewer
                        paper={loadedPaper}
                        activeCitation={selectedFact.citation}
                        apiUrl={apiUrl}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs text-amber-900">
                    <span className="font-semibold block mb-1">
                      Exact highlight unavailable
                    </span>
                    <p className="text-[11px] leading-relaxed">
                      Exact highlight unavailable — citation anchor could not be
                      verified against the current PDF document. No guessed
                      rectangle shown.
                    </p>
                  </div>
                )}

                {/* Deep Link: Open in Workspace */}
                <div className="pt-2 border-t border-zinc-200">
                  <Link
                    href={`/?project=${projectId}&paper=${selectedFact.paper_id}&page=${selectedFact.page_number}&fact=${selectedFact.id}`}
                    className="inline-flex items-center justify-center w-full rounded-lg bg-zinc-900 px-4 py-2 text-xs font-medium text-white shadow-2xs hover:bg-zinc-800 transition"
                  >
                    Open in Workspace
                  </Link>
                </div>
              </div>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
