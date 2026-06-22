import type {
  GraphFactDetail,
  GraphIndexResponse,
  GraphNeighbor,
  GraphNode,
  GraphStatus,
} from "@/types";

export class GraphError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "GraphError";
    this.status = status;
    Object.setPrototypeOf(this, GraphError.prototype);
  }
}

export class GraphUnavailableError extends GraphError {
  constructor(message = "Graph service is currently unavailable") {
    super(message, 503);
    this.name = "GraphUnavailableError";
    Object.setPrototypeOf(this, GraphUnavailableError.prototype);
  }
}

export class GraphNotFoundError extends GraphError {
  constructor(message = "Graph resource not found") {
    super(message, 404);
    this.name = "GraphNotFoundError";
    Object.setPrototypeOf(this, GraphNotFoundError.prototype);
  }
}

function buildUrl(
  apiUrl: string,
  path: string,
  params?: URLSearchParams,
): string {
  const base = apiUrl.replace(/\/+$/, "");
  const query = params && params.toString() ? `?${params.toString()}` : "";
  return `${base}${path}${query}`;
}

async function handleResponseError(res: Response): Promise<never> {
  let detailMessage: string | null = null;
  try {
    const errorBody = (await res.json()) as {
      detail?: string | Array<{ msg?: string }>;
      message?: string;
    };
    if (typeof errorBody?.detail === "string") {
      detailMessage = errorBody.detail;
    } else if (Array.isArray(errorBody?.detail)) {
      detailMessage = errorBody.detail
        .map((item) => item.msg || JSON.stringify(item))
        .join("; ");
    } else if (typeof errorBody?.message === "string") {
      detailMessage = errorBody.message;
    }
  } catch {
    // Response body was not JSON or empty
  }

  const message = detailMessage || res.statusText || `HTTP ${res.status}`;

  if (res.status === 503) {
    throw new GraphUnavailableError(message);
  }
  if (res.status === 404) {
    throw new GraphNotFoundError(message);
  }
  throw new GraphError(message, res.status);
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch (error) {
    if (error instanceof GraphError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new GraphError(`Network error: ${message}`);
  }

  if (!res.ok) {
    await handleResponseError(res);
  }

  return (await res.json()) as T;
}

export async function fetchGraphStatus(
  apiUrl: string,
  projectId: string,
): Promise<GraphStatus> {
  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/status`,
  );
  return request<GraphStatus>(url);
}

export async function searchGraphNodes(
  apiUrl: string,
  projectId: string,
  params?: {
    query?: string;
    entityType?: string;
    limit?: number;
    skip?: number;
  },
): Promise<{ items: GraphNode[]; total: number; limit: number; skip: number }> {
  const searchParams = new URLSearchParams();
  if (params?.query !== undefined && params.query.trim() !== "") {
    searchParams.set("query", params.query.trim());
  }
  if (params?.entityType !== undefined && params.entityType.trim() !== "") {
    searchParams.set("entity_type", params.entityType.trim());
  }
  if (params?.limit !== undefined) {
    searchParams.set("limit", String(params.limit));
  }
  if (params?.skip !== undefined) {
    searchParams.set("skip", String(params.skip));
  }

  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/nodes`,
    searchParams,
  );
  return request<{
    items: GraphNode[];
    total: number;
    limit: number;
    skip: number;
  }>(url);
}

export async function fetchNodeDetail(
  apiUrl: string,
  projectId: string,
  nodeKey: string,
): Promise<GraphNode> {
  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/nodes/${encodeURIComponent(nodeKey)}`,
  );
  return request<GraphNode>(url);
}

export async function fetchNodeNeighbors(
  apiUrl: string,
  projectId: string,
  nodeKey: string,
  params?: {
    direction?: string;
    predicate?: string;
    limit?: number;
  },
): Promise<{ node_key: string; neighbors: GraphNeighbor[]; total: number }> {
  const searchParams = new URLSearchParams();
  if (params?.direction !== undefined && params.direction.trim() !== "") {
    searchParams.set("direction", params.direction.trim());
  }
  if (params?.predicate !== undefined && params.predicate.trim() !== "") {
    searchParams.set("predicate", params.predicate.trim());
  }
  if (params?.limit !== undefined) {
    searchParams.set("limit", String(params.limit));
  }

  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/nodes/${encodeURIComponent(nodeKey)}/neighbors`,
    searchParams,
  );
  return request<{
    node_key: string;
    neighbors: GraphNeighbor[];
    total: number;
  }>(url);
}

export async function fetchFactDetail(
  apiUrl: string,
  projectId: string,
  factId: string,
): Promise<GraphFactDetail> {
  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/facts/${encodeURIComponent(factId)}`,
  );
  return request<GraphFactDetail>(url);
}

export async function fetchRelationships(
  apiUrl: string,
  projectId: string,
  subjectKey: string,
  objectKey: string,
  predicate?: string,
): Promise<{ items: GraphFactDetail[]; total: number }> {
  const searchParams = new URLSearchParams();
  searchParams.set("subject_key", subjectKey);
  searchParams.set("object_key", objectKey);
  if (predicate !== undefined && predicate.trim() !== "") {
    searchParams.set("predicate", predicate.trim());
  }

  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/relationships`,
    searchParams,
  );
  return request<{ items: GraphFactDetail[]; total: number }>(url);
}

export async function triggerGraphIndex(
  apiUrl: string,
  projectId: string,
  payloadOrPaperIds?:
    | {
        paperIds?: string[];
        limit?: number;
        dryRun?: boolean;
      }
    | string[],
  limit?: number,
  dryRun?: boolean,
): Promise<GraphIndexResponse> {
  const url = buildUrl(
    apiUrl,
    `/api/v1/projects/${encodeURIComponent(projectId)}/graph/index`,
  );
  const body: Record<string, unknown> = {};

  if (Array.isArray(payloadOrPaperIds)) {
    body.paper_ids = payloadOrPaperIds;
    if (limit !== undefined) {
      body.limit = limit;
    }
    if (dryRun !== undefined) {
      body.dry_run = dryRun;
    }
  } else if (payloadOrPaperIds && typeof payloadOrPaperIds === "object") {
    if (payloadOrPaperIds.paperIds !== undefined) {
      body.paper_ids = payloadOrPaperIds.paperIds;
    }
    if (payloadOrPaperIds.limit !== undefined) {
      body.limit = payloadOrPaperIds.limit;
    }
    if (payloadOrPaperIds.dryRun !== undefined) {
      body.dry_run = payloadOrPaperIds.dryRun;
    }
  } else {
    if (limit !== undefined) {
      body.limit = limit;
    }
    if (dryRun !== undefined) {
      body.dry_run = dryRun;
    }
  }

  return request<GraphIndexResponse>(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}
