from dataclasses import dataclass

from app.ingestion.parser import ParsedElement


@dataclass
class ChunkSpec:
    chunk_type: str  # "parent" or "child"
    chunk_index: int
    text: str
    token_count: int
    element_indices: list[int]  # mapped to element_index


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


class DocumentChunker:
    """Structure-aware chunker producing parent and child chunks with source element links."""

    def __init__(
        self,
        child_min_tokens: int = 50,
        child_max_tokens: int = 400,
        parent_max_tokens: int = 1500,
    ) -> None:
        self.child_min_tokens = child_min_tokens
        self.child_max_tokens = child_max_tokens
        self.parent_max_tokens = parent_max_tokens

    def chunk(self, elements: list[ParsedElement]) -> list[ChunkSpec]:
        if not elements:
            return []

        chunks: list[ChunkSpec] = []
        child_idx = 0
        parent_idx = 0

        current_child_elements: list[ParsedElement] = []
        current_child_tokens = 0

        child_chunks: list[ChunkSpec] = []

        for elem in elements:
            elem_tokens = estimate_tokens(elem.text)
            page_changed = (
                current_child_elements
                and elem.page_number != current_child_elements[-1].page_number
            )
            exceeds_tokens = current_child_tokens + elem_tokens > self.child_max_tokens
            if (exceeds_tokens or page_changed) and current_child_elements:
                # Flush child chunk
                child_text = "\n\n".join(e.text for e in current_child_elements)
                child = ChunkSpec(
                    chunk_type="child",
                    chunk_index=child_idx,
                    text=child_text,
                    token_count=current_child_tokens,
                    element_indices=[e.element_index for e in current_child_elements],
                )
                child_chunks.append(child)
                chunks.append(child)
                child_idx += 1

                current_child_elements = [elem]
                current_child_tokens = elem_tokens
            else:
                current_child_elements.append(elem)
                current_child_tokens += elem_tokens

        if current_child_elements:
            child_text = "\n\n".join(e.text for e in current_child_elements)
            child = ChunkSpec(
                chunk_type="child",
                chunk_index=child_idx,
                text=child_text,
                token_count=current_child_tokens,
                element_indices=[e.element_index for e in current_child_elements],
            )
            child_chunks.append(child)
            chunks.append(child)

        # Build parent chunks from children
        current_parent_children: list[ChunkSpec] = []
        current_parent_tokens = 0

        for child in child_chunks:
            if (
                current_parent_tokens + child.token_count > self.parent_max_tokens
                and current_parent_children
            ):
                parent_text = "\n\n".join(c.text for c in current_parent_children)
                all_element_indices = []
                for c in current_parent_children:
                    all_element_indices.extend(c.element_indices)
                parent = ChunkSpec(
                    chunk_type="parent",
                    chunk_index=parent_idx,
                    text=parent_text,
                    token_count=current_parent_tokens,
                    element_indices=list(dict.fromkeys(all_element_indices)),
                )
                chunks.append(parent)
                parent_idx += 1

                current_parent_children = [child]
                current_parent_tokens = child.token_count
            else:
                current_parent_children.append(child)
                current_parent_tokens += child.token_count

        if current_parent_children:
            parent_text = "\n\n".join(c.text for c in current_parent_children)
            all_element_indices = []
            for c in current_parent_children:
                all_element_indices.extend(c.element_indices)
            parent = ChunkSpec(
                chunk_type="parent",
                chunk_index=parent_idx,
                text=parent_text,
                token_count=current_parent_tokens,
                element_indices=list(dict.fromkeys(all_element_indices)),
            )
            chunks.append(parent)

        return chunks
