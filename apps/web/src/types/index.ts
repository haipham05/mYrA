export interface Project {
  id: string;
  name: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export type PaperStatus = "PROCESSING" | "READY" | "FAILED";

export interface Paper {
  id: string;
  project_id: string;
  filename: string;
  status: PaperStatus;
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
  created_at: string;
}

export interface Conversation {
  id: string;
  project_id: string;
  title?: string | null;
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
