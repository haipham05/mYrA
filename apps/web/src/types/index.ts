export interface Project {
  id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ProviderBudgetUsage {
  status: "available" | "unavailable";
  runtime_profile: string;
  unavailable_reason?: string | null;
  currency: "USD";
  daily_limit_estimate_usd: string;
  daily_remaining_estimate_usd: string | null;
  daily: {
    committed_estimate_usd: string;
    active_reservation_usd: string;
    unknown_reservation_usd: string;
    reported_prompt_tokens: number | null;
    reported_completion_tokens: number | null;
  };
  usage_note: string;
}

export type PaperStatus = "PROCESSING" | "READY" | "FAILED";

export interface IngestionJobSummary {
  id: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  stage: string;
  progress: number;
  error_message?: string | null;
  is_retryable: boolean;
  retry_count: number;
}

export interface Paper {
  id: string;
  project_id: string;
  filename: string;
  status: PaperStatus;
  latest_job?: IngestionJobSummary | null;
  title?: string | null;
  authors?: string[] | null;
  publication_year?: number | null;
  doi?: string | null;
  arxiv_id?: string | null;
  abstract?: string | null;
  source_url?: string | null;
  metadata_provenance?: Record<string, string> | null;
  page_count?: number | null;
  error_message?: string | null;
  document_sha256?: string | null;
  created_at: string;
  updated_at: string;
}

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  page_width: number;
  page_height: number;
  origin?: "TOP_LEFT" | "BOTTOM_LEFT";
  rotation?: number;
}

export type AnchorStatus = "verified" | "unresolved" | "legacy";

export interface CitationAnchor {
  id: string;
  page_number: number;
  source_element_id?: string | null;
  exact_quote: string;
  source_char_start?: number | null;
  source_char_end?: number | null;
  document_sha256?: string | null;
  parser_version?: string | null;
  anchor_status: AnchorStatus;
  bounding_boxes: BoundingBox[];
}

export interface Citation {
  citation_index: number;
  evidence_id: string;
  paper_id: string;
  page_number: number;
  bounding_boxes: BoundingBox[];
  quote: string;
  document_sha256?: string | null;
  parser_version?: string | null;
  anchor_status?: AnchorStatus;
  anchors?: CitationAnchor[];
}

export interface SourceSelection {
  paper_id: string;
  page_number: number;
  quote: string;
  document_sha256: string;
}

export interface VisualSelection {
  paper_id: string;
  page_number: number;
  document_sha256: string;
  crop: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
}

export interface VisualSourceReference {
  source_kind: "visual";
  project_id: string;
  paper_id: string;
  document_sha256: string;
  page_number: number;
  crop_sha256: string;
  crop_box_normalized_top_left: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
  caption?: string | null;
  text_citation: false;
}

export interface EvidenceItem {
  id: string;
  paper_id: string;
  paper_title?: string | null;
  chunk_id: string;
  quote: string;
  page_number: number;
  bounding_boxes: BoundingBox[];
  source_element_ids: string[];
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "USER" | "ASSISTANT";
  content: string;
  citations: Citation[];
  evidence: EvidenceItem[];
  model_name?: string | null;
  token_count?: number | null;
  assistantResult?: AssistantRunResult;
  created_at: string;
}

export type AssistantIntent =
  | "help"
  | "qa"
  | "read_paper"
  | "compare"
  | "verify_claim"
  | "discover"
  | "notes"
  | "report"
  | "research"
  | "gap_analysis"
  | "experiment_plan"
  | "translate"
  | "vision"
  | "graph"
  | "clarify";

export type AssistantResultType =
  | "answer"
  | "reading_brief"
  | "comparison"
  | "claim_verification"
  | "discovery_results"
  | "discovery_unavailable"
  | "research_draft"
  | "research_report"
  | "report_evidence_unavailable"
  | "gap_analysis"
  | "experiment_proposal"
  | "visual_analysis"
  | "visual_analysis_unavailable"
  | "translation_job"
  | "graph_index_jobs"
  | "graph_index_approval_required"
  | "graph_index_approval_invalid"
  | "graph_indexing_unavailable"
  | "notes_list"
  | "note"
  | "note_not_found"
  | "note_source_unavailable"
  | "note_version_conflict"
  | "note_update_rejected"
  | "clarification"
  | "approval_required"
  | "approval_invalidated"
  | "routing_unavailable"
  | "tool_error"
  | "action_outcome_unknown"
  | "source_selection_unavailable"
  | "visual_selection_required"
  | "unavailable"
  | "help"
  | "comparison_scope_unavailable"
  | "claim_scope_unavailable"
  | "research_evidence_unavailable"
  | "gap_analysis_evidence_unavailable"
  | "experiment_evidence_unavailable";

export interface AssistantResultPayloadMap {
  answer: {
    [key: string]: unknown;
    message_id?: unknown;
    model_name?: unknown;
  };
  reading_brief: {
    [key: string]: unknown;
    message_id?: unknown;
    model_name?: unknown;
    sections?: unknown;
  };
  comparison: Record<string, unknown>;
  claim_verification: {
    [key: string]: unknown;
    claim?: unknown;
    verdict?: unknown;
    explanation?: unknown;
    scope_note?: unknown;
  };
  discovery_results: Record<string, unknown>;
  discovery_unavailable: Record<string, unknown>;
  research_draft: Record<string, unknown>;
  research_report: {
    [key: string]: unknown;
    report_markdown?: unknown;
    scope?: unknown;
    source_manifest?: unknown;
    saved?: unknown;
  };
  report_evidence_unavailable: Record<string, unknown>;
  gap_analysis: Record<string, unknown>;
  experiment_proposal: Record<string, unknown>;
  visual_analysis: Record<string, unknown>;
  visual_analysis_unavailable: Record<string, unknown>;
  translation_job: Record<string, unknown>;
  graph_index_jobs: {
    [key: string]: unknown;
    project_id?: unknown;
    paper_ids?: unknown;
    queued_count?: unknown;
    skipped_count?: unknown;
    completed_count?: unknown;
    failed_count?: unknown;
    approval_id?: unknown;
  };
  graph_index_approval_required: Record<string, unknown>;
  graph_index_approval_invalid: Record<string, unknown>;
  graph_indexing_unavailable: Record<string, unknown>;
  notes_list: { [key: string]: unknown; items?: unknown; total?: unknown };
  note: { [key: string]: unknown; item?: unknown };
  note_not_found: Record<string, unknown>;
  note_source_unavailable: Record<string, unknown>;
  note_version_conflict: Record<string, unknown>;
  note_update_rejected: Record<string, unknown>;
  clarification: { [key: string]: unknown; missing_information?: unknown };
  approval_required: Record<string, unknown>;
  approval_invalidated: Record<string, unknown>;
  routing_unavailable: Record<string, unknown>;
  tool_error: Record<string, unknown>;
  action_outcome_unknown: Record<string, unknown>;
  source_selection_unavailable: Record<string, unknown>;
  visual_selection_required: Record<string, unknown>;
  unavailable: Record<string, unknown>;
  help: Record<string, unknown>;
  comparison_scope_unavailable: Record<string, unknown>;
  claim_scope_unavailable: Record<string, unknown>;
  research_evidence_unavailable: Record<string, unknown>;
  gap_analysis_evidence_unavailable: Record<string, unknown>;
  experiment_evidence_unavailable: Record<string, unknown>;
}

interface AssistantRunResultBase {
  display_text: string;
  citations: Citation[];
  warnings: string[];
  usage: Record<string, number | string | null>;
  available_actions: string[];
  artifact_ids: string[];
}

/** Known result tags carry a typed payload; unknown server tags are handled at runtime as text. */
export type AssistantRunResult = {
  [K in AssistantResultType]: AssistantRunResultBase & {
    result_type: K;
    structured_payload: AssistantResultPayloadMap[K];
  };
}[AssistantResultType];

export interface AssistantRunResponse {
  id: string;
  project_id: string;
  conversation_id: string;
  status:
    | "QUEUED"
    | "RUNNING"
    | "NEEDS_INPUT"
    | "AWAITING_APPROVAL"
    | "SUCCEEDED"
    | "FAILED"
    | "CANCELLED";
  intent: AssistantIntent | null;
  action_summary: string | null;
  stage: string | null;
  result: AssistantRunResult | null;
  safe_error: string | null;
  usage: Record<string, number | string | null> | null;
  created_at: string;
  updated_at: string;
}

export interface AssistantApprovalResponse {
  id: string;
  run_id: string;
  action_type: string;
  arguments: Record<string, unknown>;
  source_fingerprint: string;
  status: string;
  expires_at: string;
  decided_at: string | null;
  import_result?: {
    paper_id: string;
    job_id: string;
    status: PaperStatus;
  } | null;
}

export interface Conversation {
  id: string;
  project_id: string;
  title?: string | null;
  summary?: string | null;
  paper_scope?: "paper" | "selection" | "project";
  selected_paper_ids?: string[];
  is_archived?: boolean;
  message_count?: number;
  created_at: string;
  updated_at: string;
}

export interface Job {
  id: string;
  paper_id: string;
  status: "PENDING" | "PROCESSING" | "COMPLETED" | "FAILED";
  stage: string;
  progress: number;
  error_message?: string | null;
}

export type MemoryType =
  | "DECISION"
  | "PREFERENCE"
  | "TERMINOLOGY"
  | "PROCEDURAL"
  | "PAPER_FACT"
  | "EPISODIC";

export type MemoryStatus = "ACTIVE" | "SUPERSEDED" | "ARCHIVED" | "DISPUTED";
export type MemorySourceType = "MESSAGE" | "PAPER_CHUNK";

export interface MemorySource {
  id: string;
  memory_id: string;
  source_type: MemorySourceType;
  message_id?: string | null;
  conversation_id?: string | null;
  paper_id?: string | null;
  page_number?: number | null;
  quote_text?: string | null;
  document_sha256?: string | null;
  chunk_id?: string | null;
  source_element_id?: string | null;
  source_char_start?: number | null;
  source_char_end?: number | null;
  parser_version?: string | null;
  anchor_status?: AnchorStatus | null;
  bounding_boxes?: BoundingBox[];
  anchors?: CitationAnchor[];
  created_at: string;
}

export interface MemoryAudit {
  id: string;
  memory_id: string;
  action: string;
  old_content?: string | null;
  new_content?: string | null;
  reason?: string | null;
  created_at: string;
}

export interface Memory {
  id: string;
  project_id: string;
  memory_type: MemoryType;
  status: MemoryStatus;
  title: string;
  content: string;
  confidence: number;
  importance: number;
  version: number;
  is_pinned: boolean;
  superseded_by_id?: string | null;
  created_at: string;
  updated_at: string;
  last_accessed_at?: string | null;
  sources: MemorySource[];
  history: MemoryAudit[];
}

export type EntityType =
  | "Paper"
  | "Author"
  | "Institution"
  | "Task"
  | "Method"
  | "Model"
  | "Dataset"
  | "Metric"
  | "Result"
  | "Claim"
  | "Limitation"
  | "Concept";

export type RelationshipPredicate =
  | "EVALUATED_ON"
  | "ACHIEVES_RESULT"
  | "PROPOSES_METHOD"
  | "USES_MODEL"
  | "CONTRADICTS"
  | "EXTENDS"
  | "AUTHORED_BY"
  | "AFFILIATED_WITH";

export interface GraphNode {
  key: string;
  project_id: string;
  name: string;
  type: string;
  description?: string | null;
  aliases: string[];
  updated_at?: string | null;
}

export interface GraphNeighbor {
  neighbor_key: string;
  neighbor_name: string;
  neighbor_type: string;
  direction: "OUTGOING" | "INCOMING" | "BOTH";
  predicate: string;
  fact_id: string;
  qualifiers?: Record<string, unknown> | null;
}

export interface GraphFactDetail {
  id: string;
  project_id: string;
  paper_id: string;
  paper_title?: string | null;
  generation_id: string;
  predicate: string;
  subject_key: string;
  subject_name: string;
  subject_type: string;
  object_key: string;
  object_name: string;
  object_type: string;
  qualifiers?: Record<string, unknown> | null;
  exact_quote: string;
  page_number: number;
  char_start: number;
  char_end: number;
  document_sha256: string;
  updated_at?: string | null;
  citation?: Citation | null;
  anchor_status: AnchorStatus;
}

export interface GraphStatus {
  project_id: string;
  graphrag_enabled: boolean;
  neo4j_available: boolean;
  node_count: number;
  fact_count: number;
  pending_events_count: number;
  completed_events_count: number;
  failed_events_count: number;
}

export interface GraphIndexResponse {
  dry_run: boolean;
  eligible_paper_ids: string[];
  enqueued_count: number;
  skipped_count: number;
  target_project_id: string;
}
