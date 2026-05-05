"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [apiStatus, setApiStatus] = useState("Checking API…");

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

    fetch(`${apiUrl}/health`)
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then(() => setApiStatus("Connected"))
      .catch(() => setApiStatus("Unavailable"));
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center bg-zinc-50 p-8 font-sans text-zinc-900">
      <section className="w-full max-w-xl rounded-xl bg-white p-10 shadow-sm">
        <p className="text-sm font-medium text-zinc-500">My Research Assistant</p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">mYrA</h1>
        <p className="mt-6 text-zinc-600">
          API: <span className="font-medium text-zinc-900">{apiStatus}</span>
        </p>
      </section>
    </main>
  );
}
