import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import Home from "./page";

describe("Home", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ status: "ok" }) }),
    );
  });

  it("renders the application name", () => {
    render(<Home />);

    expect(screen.getByRole("heading", { name: "mYrA" })).toBeInTheDocument();
  });
});
