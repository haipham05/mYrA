import { test, expect } from "@playwright/test";

test.describe("Milestone 4 — Long-Term Research Memory Journey", () => {
  const project1 = {
    id: "proj-1111-1111-1111-111111111111",
    name: "Autonomous Driving Calibration",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  const project2 = {
    id: "proj-2222-2222-2222-222222222222",
    name: "Medical Image Analysis",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  const initialMemories = [
    {
      id: "mem-aurc-1",
      project_id: project1.id,
      memory_type: "DECISION",
      status: "ACTIVE",
      title: "Decision: Choose AURC over ECE",
      content:
        "Project decision: Selected AURC over ECE for calibration evaluation.",
      confidence: 0.95,
      importance: 0.9,
      version: 1,
      is_pinned: true,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      sources: [
        {
          id: "src-1",
          memory_id: "mem-aurc-1",
          source_type: "MESSAGE",
          message_id: "msg-1",
          created_at: new Date().toISOString(),
        },
      ],
      history: [
        {
          id: "aud-1",
          memory_id: "mem-aurc-1",
          action: "CREATED",
          new_content:
            "Project decision: Selected AURC over ECE for calibration evaluation.",
          reason: "Initial choice",
          created_at: new Date().toISOString(),
        },
      ],
    },
    {
      id: "mem-paper-1",
      project_id: project1.id,
      memory_type: "PAPER_FACT",
      status: "ACTIVE",
      title: "Attention Mechanism Fact",
      content:
        "The Transformer model uses multi-head attention mechanism across sub-layers.",
      confidence: 1.0,
      importance: 0.9,
      version: 1,
      is_pinned: false,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      sources: [
        {
          id: "src-paper-1",
          memory_id: "mem-paper-1",
          source_type: "PAPER_CHUNK",
          paper_id: "paper-1111-1111-1111-111111111111",
          page_number: 1,
          quote_text:
            "The Transformer model uses multi-head attention mechanism across sub-layers.",
          document_sha256: "hash_attention_123",
          parser_version: "docling_test",
          anchor_status: "verified",
          bounding_boxes: [],
          anchors: [
            {
              id: "anchor-1",
              page_number: 1,
              source_element_id: "elem-1",
              exact_quote:
                "The Transformer model uses multi-head attention mechanism across sub-layers.",
              source_char_start: 0,
              source_char_end: 76,
              document_sha256: "hash_attention_123",
              parser_version: "docling_test",
              anchor_status: "verified",
              bounding_boxes: [],
            },
          ],
          created_at: new Date().toISOString(),
        },
      ],
      history: [],
    },
  ];

  const mockPaper1 = {
    id: "paper-1111-1111-1111-111111111111",
    project_id: project1.id,
    filename: "attention.pdf",
    status: "READY",
    page_count: 5,
    document_sha256: "hash_attention_123",
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  let currentMemories = [...initialMemories];

  test.beforeEach(async ({ page }) => {
    currentMemories = [...initialMemories];

    // Mock API endpoints
    await page.route("**/health", async (route) => {
      await route.fulfill({ status: 200, json: { status: "ok" } });
    });

    await page.route("**/api/v1/system/status", async (route) => {
      await route.fulfill({
        status: 200,
        json: { status: "ok", deepseek_configured: true },
      });
    });

    await page.route("**/api/v1/projects", async (route) => {
      await route.fulfill({
        status: 200,
        json: { items: [project1, project2], total: 2 },
      });
    });

    await page.route(`**/api/v1/projects/${project1.id}`, async (route) => {
      await route.fulfill({ status: 200, json: project1 });
    });

    await page.route(`**/api/v1/projects/${project2.id}`, async (route) => {
      await route.fulfill({ status: 200, json: project2 });
    });

    await page.route("**/api/v1/projects/*/papers*", async (route) => {
      const url = route.request().url();
      if (url.includes(project1.id)) {
        await route.fulfill({
          status: 200,
          json: { items: [mockPaper1], total: 1 },
        });
      } else {
        await route.fulfill({ status: 200, json: { items: [], total: 0 } });
      }
    });

    await page.route("**/api/v1/projects/*/conversations*", async (route) => {
      await route.fulfill({ status: 200, json: { items: [], total: 0 } });
    });

    // Mock Memory Endpoints
    await page.route(
      `**/api/v1/projects/${project1.id}/memories**`,
      async (route) => {
        const method = route.request().method();
        const url = route.request().url();

        if (method === "GET") {
          await route.fulfill({
            status: 200,
            json: { items: currentMemories, total: currentMemories.length },
          });
        } else if (method === "POST" && url.includes("/supersede")) {
          const urlObj = new URL(url);
          const expectedVersion = urlObj.searchParams.get("expected_version");
          if (!expectedVersion) {
            await route.fulfill({
              status: 422,
              json: {
                detail: [
                  {
                    loc: ["query", "expected_version"],
                    msg: "Field required",
                    type: "missing",
                  },
                ],
              },
            });
            return;
          }
          if (expectedVersion !== "1") {
            await route.fulfill({
              status: 409,
              json: {
                detail: `MemoryVersionConflictError: Memory has current version 1, but expected_version was ${expectedVersion}`,
              },
            });
            return;
          }
          const body = route.request().postDataJSON();
          const newMem = {
            id: "mem-brier-2",
            project_id: project1.id,
            memory_type: body.memory_type || "DECISION",
            status: "ACTIVE",
            title: body.title,
            content: body.content,
            confidence: 1.0,
            importance: body.importance || 0.9,
            version: 1,
            is_pinned: false,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            sources: [],
            history: [],
          };
          // Mark old as superseded
          currentMemories = currentMemories.map((m) =>
            m.id === "mem-aurc-1"
              ? {
                  ...m,
                  status: "SUPERSEDED",
                  superseded_by_id: newMem.id,
                  version: 2,
                }
              : m,
          );
          currentMemories.push(newMem);
          await route.fulfill({ status: 200, json: newMem });
        } else {
          await route.fulfill({ status: 200, json: {} });
        }
      },
    );

    await page.route(
      `**/api/v1/projects/${project2.id}/memories**`,
      async (route) => {
        await route.fulfill({ status: 200, json: { items: [], total: 0 } });
      },
    );
  });

  test("user can inspect active decision, supersede it, and verify project scope", async ({
    page,
  }) => {
    await page.goto("http://127.0.0.1:3000");

    // 1. Verify Project 1 is selected
    const projectSelect = page.locator("select").first();
    await expect(projectSelect).toHaveValue(project1.id);

    // 2. Click "Project Memory" tab
    const memoryTab = page.getByRole("button", { name: "Project Memory" });
    await memoryTab.click();

    // 3. Inspect Memory Inspector header and card
    const inspector = page.getByTestId("memory-inspector");
    await expect(inspector).toBeVisible();
    const aurcCard = inspector.getByTestId("memory-card-mem-aurc-1");
    await expect(
      aurcCard.getByText("Decision: Choose AURC over ECE"),
    ).toBeVisible();
    await expect(aurcCard.getByText("DECISION", { exact: true })).toBeVisible();
    await expect(aurcCard.getByText("ACTIVE", { exact: true })).toBeVisible();
    await expect(aurcCard.getByText("v1", { exact: true })).toBeVisible();

    // 4. Supersede decision
    const supersedeBtn = aurcCard.getByRole("button", { name: "Supersede" });
    await supersedeBtn.click();

    // Fill supersede form
    const modal = page.locator("form");
    await expect(modal).toBeVisible();
    await modal.locator("input").first().fill("Switch to Brier Score");
    await modal
      .locator("textarea")
      .fill("Decided to switch from AURC to Brier Score.");
    await modal.getByRole("button", { name: "Supersede Decision" }).click();

    // 5. Verify the new active decision appears and old is superseded
    await expect(inspector.getByText("Switch to Brier Score")).toBeVisible();
    await expect(
      inspector.getByText("SUPERSEDED", { exact: true }),
    ).toBeVisible();

    // 6. Test direct route /projects/[id]/memory
    await page.goto(`http://127.0.0.1:3000/projects/${project1.id}/memory`);
    await expect(
      page.getByText("Autonomous Driving Calibration — Memory Inspector"),
    ).toBeVisible();
    await expect(page.getByTestId("memory-inspector")).toBeVisible();
  });

  test("user can jump from paper fact memory source to verified PDF citation in workspace", async ({
    page,
  }) => {
    await page.goto("http://127.0.0.1:3000");

    // 1. Verify Project 1 is selected
    const projectSelect = page.locator("select").first();
    await expect(projectSelect).toHaveValue(project1.id);

    // 2. Click "Project Memory" tab
    const memoryTab = page.getByRole("button", { name: "Project Memory" });
    await memoryTab.click();

    // 3. Inspect Memory Inspector for paper fact card
    const inspector = page.getByTestId("memory-inspector");
    await expect(inspector).toBeVisible();
    await expect(inspector.getByText("Attention Mechanism Fact")).toBeVisible();
    await expect(
      inspector.getByText("PAPER_FACT", { exact: true }),
    ).toBeVisible();

    // 4. Click "View Paper Source (p. 1)" button
    const viewSourceBtn = inspector.getByRole("button", {
      name: "View Paper Source (p. 1)",
    });
    await expect(viewSourceBtn).toBeVisible();
    await viewSourceBtn.click();

    // 5. Verify UI switched to workspace tab
    const workspaceTab = page.getByRole("button", { name: "Workspace" });
    await expect(workspaceTab).toHaveClass(/bg-white/);

    // 6. Verify PDF viewer shows active citation callout with exact quote
    await expect(page.getByText("Cited on Page 1")).toBeVisible();
    await expect(
      page.getByText(
        "The Transformer model uses multi-head attention mechanism across sub-layers.",
      ),
    ).toBeVisible();
  });
});
