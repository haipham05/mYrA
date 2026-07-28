import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";

const fixturePath = path.resolve(
  __dirname,
  "../../../tests/retrieval_eval/fixtures/devlin2018_bert.pdf",
);
const fixturePdf = fs.readFileSync(fixturePath);
const documentHash = createHash("sha256").update(fixturePdf).digest("hex");
const projectId = "11111111-1111-4111-8111-111111111111";
const firstPaperId = "22222222-2222-4222-8222-222222222222";
const secondPaperId = "33333333-3333-4333-8333-333333333333";
const conversationId = "44444444-4444-4444-8444-444444444444";
const quote = "On the GLUE benchmark, BERT achieves an overall score of 80.5%,";

const papers = [
  {
    id: firstPaperId,
    project_id: projectId,
    filename: "bert-baseline.pdf",
    title: "BERT Baseline Study",
    status: "READY",
    page_count: 15,
    document_sha256: documentHash,
    created_at: "2026-10-05T00:00:00Z",
    updated_at: "2026-10-05T00:00:00Z",
  },
  {
    id: secondPaperId,
    project_id: projectId,
    filename: "bert-comparison.pdf",
    title: "BERT Comparison Study",
    status: "READY",
    page_count: 15,
    document_sha256: documentHash,
    created_at: "2026-10-05T00:00:00Z",
    updated_at: "2026-10-05T00:00:00Z",
  },
];

function citation(paperId: string, index: number, evidenceId: string) {
  return {
    citation_index: index,
    evidence_id: evidenceId,
    paper_id: paperId,
    page_number: 5,
    bounding_boxes: [],
    quote,
    document_sha256: documentHash,
    parser_version: "docling-fixture-v1",
    anchor_status: "verified",
    anchors: [
      {
        id: `${evidenceId}-anchor`,
        page_number: 5,
        source_element_id: `${evidenceId}-element`,
        exact_quote: quote,
        source_char_start: 0,
        source_char_end: quote.length,
        document_sha256: documentHash,
        parser_version: "docling-fixture-v1",
        anchor_status: "verified",
        bounding_boxes: [],
      },
    ],
  };
}

test("compare two selected papers and open both cited sources", async ({
  page,
}) => {
  let submittedRunBody: Record<string, unknown> | null = null;
  let conversationScope: Record<string, unknown> = {
    paper_scope: "project",
    selected_paper_ids: [],
  };

  const firstCitation = citation(firstPaperId, 1, "evidence-bert-baseline");
  const secondCitation = citation(secondPaperId, 2, "evidence-bert-comparison");
  const comparisonResult = {
    result_type: "comparison",
    display_text: "Evidence from both selected papers is ready to compare.",
    structured_payload: {
      matrix: {
        project_id: projectId,
        paper_ids: [firstPaperId, secondPaperId],
        dimensions: ["results"],
        cells: [
          {
            paper_id: firstPaperId,
            dimension: "results",
            status: "evidence_available",
            excerpts: [
              {
                evidence: {
                  id: "evidence-bert-baseline",
                  paper_id: firstPaperId,
                  paper_title: papers[0].title,
                  chunk_id: "chunk-baseline",
                  quote,
                  page_number: 5,
                  bounding_boxes: [],
                  source_element_ids: ["baseline-element"],
                },
                citation: firstCitation,
              },
            ],
          },
          {
            paper_id: secondPaperId,
            dimension: "results",
            status: "evidence_available",
            excerpts: [
              {
                evidence: {
                  id: "evidence-bert-comparison",
                  paper_id: secondPaperId,
                  paper_title: papers[1].title,
                  chunk_id: "chunk-comparison",
                  quote,
                  page_number: 5,
                  bounding_boxes: [],
                  source_element_ids: ["comparison-element"],
                },
                citation: secondCitation,
              },
            ],
          },
        ],
      },
    },
    citations: [firstCitation, secondCitation],
    warnings: [],
    usage: {},
    available_actions: [],
    artifact_ids: [],
  };

  await page.route("**/health", (route) =>
    route.fulfill({ status: 200, json: { status: "ok" } }),
  );
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const { pathname } = url;

    if (pathname.endsWith("/system/status")) {
      await route.fulfill({
        status: 200,
        json: { status: "ok", deepseek_configured: false },
      });
    } else if (pathname.endsWith("/budget/usage")) {
      await route.fulfill({ status: 200, json: { status: "unavailable" } });
    } else if (pathname === "/api/v1/projects") {
      await route.fulfill({
        status: 200,
        json: {
          items: [
            {
              id: projectId,
              name: "Comparison fixture",
              created_at: papers[0].created_at,
              updated_at: papers[0].updated_at,
            },
          ],
          total: 1,
        },
      });
    } else if (pathname === `/api/v1/projects/${projectId}/papers`) {
      await route.fulfill({ status: 200, json: { items: papers, total: 2 } });
    } else if (pathname === `/api/v1/projects/${projectId}/conversations`) {
      if (request.method() === "POST") {
        await route.fulfill({
          status: 200,
          json: {
            id: conversationId,
            project_id: projectId,
            title: "Workspace Chat",
            paper_scope: "project",
            selected_paper_ids: [],
            is_archived: false,
            message_count: 0,
            created_at: papers[0].created_at,
            updated_at: papers[0].updated_at,
          },
        });
      } else {
        await route.fulfill({ status: 200, json: { items: [], total: 0 } });
      }
    } else if (pathname === `/api/v1/conversations/${conversationId}`) {
      conversationScope = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        json: {
          id: conversationId,
          project_id: projectId,
          title: "Workspace Chat",
          ...conversationScope,
          created_at: papers[0].created_at,
          updated_at: papers[0].updated_at,
        },
      });
    } else if (
      pathname === `/api/v1/conversations/${conversationId}/messages`
    ) {
      await route.fulfill({ status: 200, json: [] });
    } else if (pathname.endsWith("/translation-glossary")) {
      await route.fulfill({ status: 200, json: [] });
    } else if (pathname.endsWith("/translations")) {
      await route.fulfill({ status: 200, json: { items: [] } });
    } else if (
      pathname === `/api/v1/conversations/${conversationId}/runs` &&
      request.method() === "POST"
    ) {
      submittedRunBody = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        json: {
          id: "55555555-5555-4555-8555-555555555555",
          project_id: projectId,
          conversation_id: conversationId,
          status: "SUCCEEDED",
          intent: "compare",
          action_summary: "Compare selected papers",
          stage: "completed",
          result: comparisonResult,
          safe_error: null,
          usage: {},
          created_at: papers[0].created_at,
          updated_at: papers[0].updated_at,
        },
      });
    } else if (/^\/api\/v1\/papers\/[0-9a-f-]+\/document$/.test(pathname)) {
      await route.fulfill({
        status: 200,
        contentType: "application/pdf",
        body: fixturePdf,
      });
    } else {
      await route.fulfill({ status: 404, json: { detail: "Not found" } });
    }
  });

  await page.goto("/");
  await expect(page.getByText("Connected", { exact: true })).toBeVisible();
  await expect(page.locator("#conversation-select")).toHaveValue(
    conversationId,
  );
  await page.locator("#paper-scope").selectOption("selection");
  await expect(
    page.getByText("0 papers selected.", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("checkbox", {
      name: "Include BERT Baseline Study in chat scope",
    })
    .click();
  await expect(
    page.getByText("1 paper selected.", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("checkbox", {
      name: "Include BERT Comparison Study in chat scope",
    })
    .click();
  await expect(
    page.getByText("2 papers selected.", { exact: true }),
  ).toBeVisible();

  await page
    .getByPlaceholder("Ask a grounded research question…")
    .fill("Compare the reported results in these two papers.");
  await page.getByRole("button", { name: "Send", exact: true }).click();

  await expect(
    page.getByRole("row", { name: /BERT Baseline Study/ }),
  ).toBeVisible();
  await expect(
    page.getByRole("row", { name: /BERT Comparison Study/ }),
  ).toBeVisible();
  await expect(page.getByText(quote, { exact: true }).first()).toBeVisible();
  expect(conversationScope).toMatchObject({
    paper_scope: "selection",
    selected_paper_ids: [firstPaperId, secondPaperId],
  });
  expect(submittedRunBody).toMatchObject({
    scope: "selection",
    selected_paper_ids: [firstPaperId, secondPaperId],
  });

  const baselineSource = page.getByRole("button", { name: "Page 5 ·[1]" });
  await baselineSource.click();
  await expect(
    page.getByRole("heading", { name: "bert-baseline.pdf" }),
  ).toBeVisible();
  await expect(page.getByText(/Page 5 of 15/)).toBeVisible();
  await expect(page.getByText("Verbatim match")).toBeVisible();

  const comparisonSource = page.getByRole("button", { name: "Page 5 ·[2]" });
  await comparisonSource.click();
  await expect(
    page.getByRole("heading", { name: "bert-comparison.pdf" }),
  ).toBeVisible();
  await expect(page.getByText(/Page 5 of 15/)).toBeVisible();
  await expect(page.getByText("Verbatim match")).toBeVisible();
});
