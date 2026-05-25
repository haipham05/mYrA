import {
  test,
  expect,
  type APIRequestContext,
  type Page,
} from "@playwright/test";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";

const WEB_ORIGIN = "http://127.0.0.1:3000";
const apiDir = path.resolve(__dirname, "../../api");
const fixturePath = path.resolve(
  __dirname,
  "../../../tests/retrieval_eval/fixtures/devlin2018_bert.pdf",
);
const python = path.join(apiDir, ".venv/bin/python");

let tempDirectory: string | null = null;
let apiProcess: ChildProcess | null = null;
let workerProcess: ChildProcess | null = null;
let apiOrigin: string;

async function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close();
        reject(new Error("Could not allocate a local test port"));
        return;
      }
      server.close(() => resolve(address.port));
    });
  });
}

async function stopProcess(child: ChildProcess | null): Promise<void> {
  if (!child || child.exitCode !== null) return;
  child.kill("SIGTERM");
  await Promise.race([
    new Promise<void>((resolve) => child.once("exit", () => resolve())),
    new Promise<void>((resolve) => setTimeout(resolve, 2000)),
  ]);
}

async function cleanup(): Promise<void> {
  await Promise.all([stopProcess(apiProcess), stopProcess(workerProcess)]);
  apiProcess = null;
  workerProcess = null;
  if (tempDirectory) {
    if (!path.basename(tempDirectory).startsWith("myra-e2e-")) {
      throw new Error("Refusing to remove a non-test directory");
    }
    fs.rmSync(tempDirectory, { recursive: true, force: true });
    tempDirectory = null;
  }
}

async function startIsolatedApi(request: APIRequestContext): Promise<void> {
  if (!fs.existsSync(python) || !fs.existsSync(fixturePath)) {
    throw new Error(
      "The existing API venv and checked-in PDF fixture are required",
    );
  }
  tempDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "myra-e2e-"));
  const port = await freePort();
  apiOrigin = `http://127.0.0.1:${port}`;
  const env = {
    ...process.env,
    DATABASE_URL: `sqlite:///${path.join(tempDirectory, "test.db")}`,
    GCS_BUCKET_NAME: "",
    DEEPSEEK_API_KEY: "",
    MYRA_LLM_MODE: "test",
    MYRA_USE_DOCLING: "false",
    MYRA_EMBEDDING_PROVIDER: "deterministic",
    MYRA_RERANKER_PROVIDER: "simple-lexical",
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    PYTHONPATH: apiDir,
    PYTHONDONTWRITEBYTECODE: "1",
  };
  const init = spawnSync(
    python,
    ["-c", "from app.db.session import create_tables; create_tables()"],
    { cwd: tempDirectory, env, stdio: "ignore" },
  );
  if (init.status !== 0)
    throw new Error("Could not initialize isolated test database");
  apiProcess = spawn(
    python,
    [
      "-m",
      "uvicorn",
      "app.main:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
    ],
    { cwd: tempDirectory, env, stdio: "ignore" },
  );
  workerProcess = spawn(
    python,
    ["-m", "app.worker", "--poll-interval", "0.1"],
    {
      cwd: tempDirectory,
      env,
      stdio: "ignore",
    },
  );
  for (let attempt = 0; attempt < 50; attempt++) {
    if (apiProcess.exitCode !== null || workerProcess.exitCode !== null) {
      throw new Error("Isolated API or worker exited during startup");
    }
    const health = await request.get(`${apiOrigin}/health`).catch(() => null);
    if (health?.ok()) return;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("Isolated API did not become healthy");
}

async function routeOnlyToIsolatedApi(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const url = new URL(route.request().url());
    if (url.origin === WEB_ORIGIN) {
      await route.continue();
      return;
    }
    if (url.pathname === "/health" || url.pathname.startsWith("/api/v1/")) {
      const response = await route.fetch({
        url: `${apiOrigin}${url.pathname}${url.search}`,
      });
      await route.fulfill({ response });
      return;
    }
    await route.abort();
  });
}

async function assertHighlightMatchesQuote(
  page: Page,
  quote: string,
): Promise<void> {
  const observed = await page.evaluate((targetQuote) => {
    const layer = document.querySelector<HTMLElement>(
      "[data-testid=pdf-text-layer]",
    );
    if (!layer) throw new Error("PDF text layer is missing");
    const walker = document.createTreeWalker(layer, NodeFilter.SHOW_TEXT);
    const entries: { node: Text; start: number; end: number }[] = [];
    let fullText = "";
    let current: Node | null;
    while ((current = walker.nextNode())) {
      const node = current as Text;
      const value = node.nodeValue ?? "";
      if (!value) continue;
      if (fullText && !/\s$/.test(fullText) && !/^\s/.test(value))
        fullText += " ";
      entries.push({
        node,
        start: fullText.length,
        end: fullText.length + value.length,
      });
      fullText += value;
    }
    const start = fullText.indexOf(targetQuote);
    if (start < 0 || fullText.indexOf(targetQuote, start + 1) >= 0) {
      throw new Error(
        "Expected a unique verbatim quote in the rendered text layer",
      );
    }
    const end = start + targetQuote.length;
    const first = entries.find(
      (entry) => start >= entry.start && start < entry.end,
    );
    const last = entries.find((entry) => end > entry.start && end <= entry.end);
    if (!first || !last)
      throw new Error("Quote offsets could not map to text nodes");
    const range = document.createRange();
    range.setStart(first.node, start - first.start);
    range.setEnd(last.node, end - last.start);
    const layerBox = layer.getBoundingClientRect();
    const toRelative = (rect: DOMRect) => ({
      left: rect.left - layerBox.left,
      top: rect.top - layerBox.top,
      width: rect.width,
      height: rect.height,
    });
    return {
      selected: range.toString(),
      rangeRects: Array.from(range.getClientRects())
        .filter((rect) => rect.width > 0 && rect.height > 0)
        .map(toRelative),
      highlightRects: Array.from(
        document.querySelectorAll<HTMLElement>(
          "[data-testid=evidence-highlight]",
        ),
      ).map((element) => toRelative(element.getBoundingClientRect())),
    };
  }, quote);
  const normalize = (value: string) => value.replace(/\s+/g, " ").trim();
  expect(normalize(observed.selected)).toBe(normalize(quote));
  expect(observed.highlightRects).toHaveLength(observed.rangeRects.length);
  for (const [index, rect] of observed.highlightRects.entries()) {
    const source = observed.rangeRects[index];
    expect(Math.abs(rect.left - source.left)).toBeLessThan(3);
    expect(Math.abs(rect.top - source.top)).toBeLessThan(3);
    expect(Math.abs(rect.width - source.width)).toBeLessThan(3);
    expect(Math.abs(rect.height - source.height)).toBeLessThan(3);
  }
}

test.describe("isolated citation browser regression", () => {
  let sharedProjectId: string;
  let sharedPaperId: string;
  let sharedCitationQuote: string;

  test.beforeAll(async ({ request }) => {
    try {
      await startIsolatedApi(request);
    } catch (error) {
      await cleanup();
      throw error;
    }
  });
  test.afterAll(async () => cleanup());

  test("upload, ask, and highlight only the cited source line", async ({
    page,
    request,
  }) => {
    const projectResponse = await request.post(`${apiOrigin}/api/v1/projects`, {
      data: { name: "Isolated browser evidence" },
    });
    expect(projectResponse.ok()).toBeTruthy();
    const projectId = (await projectResponse.json()).id as string;
    sharedProjectId = projectId;

    const uploadResponse = await request.post(
      `${apiOrigin}/api/v1/projects/${projectId}/papers`,
      {
        multipart: {
          file: {
            name: "devlin2018_bert.pdf",
            mimeType: "application/pdf",
            buffer: fs.readFileSync(fixturePath),
          },
        },
      },
    );
    expect(uploadResponse.ok()).toBeTruthy();
    const upload = await uploadResponse.json();
    let completed = false;
    let lastJobStatus = "unknown";
    for (let attempt = 0; attempt < 100; attempt++) {
      const jobResponse = await request.get(
        `${apiOrigin}/api/v1/jobs/${upload.job_id}`,
      );
      const job = await jobResponse.json();
      lastJobStatus = `${job.status}/${job.stage}/retry=${job.retry_count}`;
      if (job.status === "COMPLETED") {
        completed = true;
        break;
      }
      if (job.status === "FAILED")
        throw new Error("Isolated PDF ingestion failed");
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
    expect(
      completed,
      `job=${lastJobStatus}, worker_exit=${workerProcess?.exitCode}`,
    ).toBeTruthy();
    const papersResponse = await request.get(
      `${apiOrigin}/api/v1/projects/${projectId}/papers`,
    );
    const paperList = await papersResponse.json();
    expect(
      paperList.items.some(
        (paper: { id: string }) => paper.id === upload.paper_id,
      ),
    ).toBe(true);

    await routeOnlyToIsolatedApi(page);
    await page.goto(WEB_ORIGIN);
    await expect(page.getByText("Connected", { exact: true })).toBeVisible();
    await page.locator("#project-select").selectOption(projectId);
    await page
      .getByRole("button", { name: /devlin2018_bert\.pdf/ })
      .click({ timeout: 10000 });

    const answerResponse = page.waitForResponse(
      (response) =>
        response.url().includes("/messages") &&
        response.request().method() === "POST",
    );
    await page
      .getByPlaceholder(/Ask a grounded research question/i)
      .fill("What score did BERT obtain on the GLUE benchmark?");
    await page.keyboard.press("Enter");
    const answer = await (await answerResponse).json();
    expect(answer.citations).toHaveLength(1);
    const citation = answer.citations[0];
    sharedPaperId = upload.paper_id;
    sharedCitationQuote = citation.quote;
    expect(citation.paper_id).toBe(upload.paper_id);
    expect(citation.page_number).toBe(5);
    expect(citation.quote).toContain("overall score of 80.5%");

    await page.getByRole("button", { name: "[1]" }).click();
    await expect(page.getByText(/Page 5 of/)).toBeVisible();
    await expect(page.getByText("Verbatim match")).toBeVisible();
    const highlights = page.getByTestId("evidence-highlight");
    await expect(highlights.first()).toBeVisible();
    await assertHighlightMatchesQuote(page, citation.quote);
    const layer = page.getByTestId("pdf-text-layer");
    const layerBox = await layer.boundingBox();
    expect(layerBox).not.toBeNull();
    for (let index = 0; index < (await highlights.count()); index++) {
      const box = await highlights.nth(index).boundingBox();
      expect(box).not.toBeNull();
      expect(box!.height).toBeGreaterThan(0);
      expect(box!.height).toBeLessThan(60);
      expect(box!.x).toBeGreaterThanOrEqual(layerBox!.x - 3);
      expect(box!.y).toBeGreaterThanOrEqual(layerBox!.y - 3);
      expect(box!.x + box!.width).toBeLessThanOrEqual(
        layerBox!.x + layerBox!.width + 3,
      );
      expect(box!.y + box!.height).toBeLessThanOrEqual(
        layerBox!.y + layerBox!.height + 3,
      );
    }

    await page.getByTitle("Zoom In").click();
    await expect(highlights.first()).toBeVisible();
    await assertHighlightMatchesQuote(page, citation.quote);
    await page.getByTitle("Rotate 90°").click();
    await expect(highlights.first()).toBeVisible();
    await assertHighlightMatchesQuote(page, citation.quote);

    await page.getByRole("button", { name: "Previous" }).click();
    await expect(page.getByText(/Page 4 of/)).toBeVisible();
    await expect(highlights).toHaveCount(0);
  });

  test("conversation persistence: reload, reopen chat, verify citations, switch, and delete chats", async ({
    page,
    request,
  }) => {
    // 1. Route requests to isolated API and load web UI
    await routeOnlyToIsolatedApi(page);
    await page.goto(WEB_ORIGIN);
    await expect(page.getByText("Connected", { exact: true })).toBeVisible();

    // 2. Select the shared project and paper
    await page.locator("#project-select").selectOption(sharedProjectId);
    await page
      .getByRole("button", { name: /devlin2018_bert\.pdf/ })
      .click({ timeout: 10000 });

    // 3. Verify conversation and messages are reopened from DB
    await expect(
      page.getByText("What score did BERT obtain on the GLUE benchmark?"),
    ).toBeVisible({ timeout: 10000 });
    await expect(page.getByText(/overall score of 80.5%/)).toBeVisible();

    const initialConvId = await page
      .locator("#conversation-select")
      .inputValue();
    expect(initialConvId).toBeTruthy();

    // 4. Click citation chip in reopened conversation and verify highlight
    await page.getByRole("button", { name: "[1]" }).click();
    await expect(page.getByText(/Page 5 of/)).toBeVisible();
    await expect(page.getByText("Verbatim match")).toBeVisible();
    await assertHighlightMatchesQuote(page, sharedCitationQuote);

    // 5. Rename conversation in UI
    await page.getByRole("button", { name: "Rename conversation" }).click();
    const titleInput = page.getByPlaceholder("Chat title");
    await titleInput.fill("BERT GLUE Discussion");
    await page.getByRole("button", { name: "Save" }).click();
    await expect(page.locator("#conversation-select")).toContainText(
      "BERT GLUE Discussion",
    );

    // 6. Test browser page reload and verify state persists
    await page.reload();
    await expect(page.getByText("Connected", { exact: true })).toBeVisible();
    await page.locator("#project-select").selectOption(sharedProjectId);
    await page
      .getByRole("button", { name: /devlin2018_bert\.pdf/ })
      .click({ timeout: 10000 });
    await expect(page.locator("#conversation-select")).toContainText(
      "BERT GLUE Discussion",
    );
    await expect(
      page.getByText("What score did BERT obtain on the GLUE benchmark?"),
    ).toBeVisible();
    await expect(page.getByText(/overall score of 80.5%/)).toBeVisible();

    // 7. Verify citation chip still highlights after page reload
    await page.getByRole("button", { name: "[1]" }).click();
    await expect(page.getByText(/Page 5 of/)).toBeVisible();
    await expect(page.getByText("Verbatim match")).toBeVisible();
    await expect(page.getByTestId("evidence-highlight").first()).toBeVisible();
    await assertHighlightMatchesQuote(page, sharedCitationQuote);

    // 8. Create a new conversation
    await page.getByRole("button", { name: "+ New Chat" }).click();
    const newChatInput = page.getByPlaceholder("New chat title (optional)");
    await newChatInput.fill("Secondary Investigation");
    await page.getByRole("button", { name: "Start" }).click();
    await expect(page.locator("#conversation-select")).toContainText(
      "Secondary Investigation",
    );
    const newConvId = await page.locator("#conversation-select").inputValue();
    expect(newConvId).not.toBe(initialConvId);

    // Verify new chat has clean message area
    await expect(
      page.getByText("What score did BERT obtain on the GLUE benchmark?"),
    ).not.toBeVisible();

    // 9. Switch back to the previous conversation
    await page.locator("#conversation-select").selectOption(initialConvId);
    await expect(
      page.getByText("What score did BERT obtain on the GLUE benchmark?"),
    ).toBeVisible();
    await expect(page.getByText(/overall score of 80.5%/)).toBeVisible();

    // 10. Switch back to new conversation and delete it
    await page.locator("#conversation-select").selectOption(newConvId);
    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "Delete conversation" }).click();
    await expect(page.locator("#conversation-select")).not.toContainText(
      "Secondary Investigation",
    );
    // After deletion, active conversation falls back to the remaining conversation
    await expect(page.locator("#conversation-select")).toHaveValue(
      initialConvId,
    );

    // 11. Edge condition: Wrong-project isolation
    const missingProjectId = "00000000-0000-0000-0000-000000000000";
    const wrongProjectRes = await request.get(
      `${apiOrigin}/api/v1/projects/${missingProjectId}/conversations`,
    );
    expect(wrongProjectRes.status()).toBe(404);

    const missingConvId = "00000000-0000-0000-0000-000000000000";
    const wrongConvRes = await request.get(
      `${apiOrigin}/api/v1/conversations/${missingConvId}/messages`,
    );
    expect(wrongConvRes.status()).toBe(404);

    // 12. Edge condition: System configuration status (non-secret, missing provider safe)
    const systemStatusRes = await request.get(
      `${apiOrigin}/api/v1/system/status`,
    );
    expect(systemStatusRes.ok()).toBeTruthy();
    const systemStatus = await systemStatusRes.json();
    expect(systemStatus.deepseek_configured).toBe(false);
    expect(systemStatus.storage_backend).toBe("local");
    const serialized = JSON.stringify(systemStatus);
    expect(serialized).not.toContain("key");
    expect(serialized).not.toContain("secret");
  });
});
